#!/usr/bin/env python3
import os
import json
import argparse
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from peft import PeftModel, LoraConfig, get_peft_model
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_hf_dataset(config):
    """Load and prepare HF dataset"""
    logger.info(f"Loading dataset: {config['hf_dataset_name']}")
    dataset = load_dataset(config["hf_dataset_name"])
    
    train_data = dataset["train"]
    if config.get("train_limit"):
        train_data = train_data.select(range(min(config["train_limit"], len(train_data))))
    
    # Filter for samples with bbox data for Stage 2
    train_data = train_data.filter(lambda x: x.get('mask_available', False) and x.get('bbox'))
    
    logger.info(f"Loaded {len(train_data)} samples with spatial data")
    return train_data

def prepare_dataset(data, processor):
    """Prepare dataset for GRPO training"""
    def format_conversation(example):
        # Get diagnosis mapping
        dx_names = {
            'akiec': 'actinic keratosis',
            'bcc': 'basal cell carcinoma', 
            'bkl': 'benign keratosis-like lesion',
            'df': 'dermatofibroma',
            'mel': 'melanoma',
            'nv': 'melanocytic nevus',
            'vasc': 'vascular lesion'
        }
        
        diagnosis_full = dx_names.get(example['diagnosis'], example['diagnosis'])
        prompt = (
            "Analyze this skin lesion. Reply concisely using tags: "
            "<diagnosis>...</diagnosis> and <location>...</location>."
        )
        
        # Format spatial description
        spatial_desc = example['spatial_description']
        response = (
            f"<diagnosis>{diagnosis_full}</diagnosis> "
            f"<location>{spatial_desc}</location>"
        )
        
        # Create conversation format
        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": response}]}
        ]
        
        # Build minimal chat-templated prompt that includes the image token but no generation prompt
        # Keeping it short helps avoid prompt truncation without increasing max_prompt_length
        prompt_text = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        
        return {
            # GRPO expects a prompt string (including image token) and images
            "prompt": prompt_text,
            "images": example["image"],
            "chosen": response,
            # Provide explicit fields some GRPO reward functions/datasets expect
            "completion": response,
            "answer": diagnosis_full,
            "bbox": example["bbox"],
            "diagnosis": example["diagnosis"]
        }
    
    return data.map(format_conversation)

def calculate_reward(prompts, completions, **kwargs):
    """Calculate reward for GRPO training with diagnosis, spatial, and bbox components."""
    import re
    
    if isinstance(completions, str):
        completions = [completions]
    
    examples = kwargs.get('examples', [{}] * len(completions))
    
    # Reward weights
    weights = {"diagnosis": 0.5, "spatial": 0.3, "segmentation": 0.2}
    
    # Diagnosis mapping
    dx_names = {
        'akiec': 'actinic keratosis', 'bcc': 'basal cell carcinoma', 
        'bkl': 'benign keratosis-like lesion', 'df': 'dermatofibroma',
        'mel': 'melanoma', 'nv': 'melanocytic nevus', 'vasc': 'vascular lesion'
    }
    
    def extract_tag(text, tag):
        match = re.search(rf"<{tag}>([\s\S]*?)</{tag}>", text, re.IGNORECASE)
        return match.group(1).strip() if match else ""
    
    rewards = []
    for i, completion in enumerate(completions):
        example = examples[i] if i < len(examples) else {}
        reward = 0.0
        
        # Extract model predictions
        pred_diag = extract_tag(completion, "diagnosis").lower()
        pred_loc = extract_tag(completion, "location").lower()
        # Get expected diagnosis - handle both short codes and full names
        diagnosis_code = example.get('diagnosis', '')
        expected_diag = dx_names.get(diagnosis_code, diagnosis_code).lower()
        
        # 1. Diagnosis accuracy (50%)
        if expected_diag and pred_diag and (expected_diag in pred_diag or pred_diag in expected_diag):
            reward += weights["diagnosis"]
        
        # 2. Spatial location accuracy (30%)
        gt_spatial = str(example.get('spatial_description', '')).lower()
        if gt_spatial and pred_loc:
            spatial_tokens = ['center-center', 'upper-left', 'upper-right', 'lower-left', 'lower-right', 'left', 'right', 'center', 'top', 'bottom']
            if any(token in pred_loc and token in gt_spatial for token in spatial_tokens):
                reward += weights["spatial"]
        
        # 3. Bbox reasoning (20%)
        bbox = example.get('bbox', [])
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            area_coverage = float(example.get('area_coverage', 0) or 0)
            completion_lower = completion.lower()
            
            size_reward = 0.0
            # Size reasoning
            if area_coverage > 0.2 and ("large" in completion_lower or "big" in completion_lower):
                size_reward += 0.1
            elif area_coverage < 0.1 and ("small" in completion_lower or "tiny" in completion_lower):
                size_reward += 0.1
            
            # Position reasoning - use bbox ratio to estimate position
            # Since we don't have image dimensions, we can estimate based on bbox width/height ratio
            x1, y1, x2, y2 = bbox
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            
            # Estimate image dimensions from area_coverage and bbox area
            bbox_area = bbox_width * bbox_height
            if area_coverage > 0 and bbox_height > 0 and bbox_area > 0:
                estimated_img_area = bbox_area / area_coverage
                estimated_img_width = (estimated_img_area * (bbox_width / bbox_height)) ** 0.5
                
                # Normalize center position (avoid division by zero)
                if estimated_img_width > 0:
                    norm_center_x = center_x / estimated_img_width
                    
                    # Check position matches
                    if norm_center_x > 0.67 and "right" in completion_lower:
                        size_reward += 0.05
                    elif norm_center_x < 0.33 and "left" in completion_lower:
                        size_reward += 0.05
                    elif 0.33 <= norm_center_x <= 0.67 and "center" in completion_lower:
                        size_reward += 0.05
            
            reward += min(weights["segmentation"], size_reward)
        
        # Penalties for poor formatting
        if not pred_diag: reward *= 0.8
        if not pred_loc: reward *= 0.9
        if len(completion.split()) > 120: reward *= 0.85
        
        rewards.append(max(0.0, min(1.0, reward)))
    
    return rewards

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config_stage2_hf.json')
    parser.add_argument('--stage1_model', default=None, help='Override stage1 model path from config')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)
    
    # Use command line argument if provided, otherwise use config
    stage1_model_path = args.stage1_model or config.get('stage1_model_path')
    
    if not stage1_model_path:
        raise ValueError("Stage 1 model path must be specified either in config or via --stage1_model argument")
    
    logger.info(f"Loading Stage 1 model from: {stage1_model_path}")
    
    # Check if using LoRA
    use_lora = config.get("lora", {}).get("use_lora", False)
    base_model_name = config.get("base_model_name", "Qwen/Qwen2-VL-2B-Instruct")
    
    if use_lora and os.path.exists(stage1_model_path):
        logger.info("Loading LoRA-trained model...")
        # Load base model
        logger.info(f"Loading base model: {base_model_name}")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA adapter from Stage 1
        logger.info(f"Loading Stage 1 LoRA adapter from: {stage1_model_path}")
        model = PeftModel.from_pretrained(model, stage1_model_path)
        
        # DON'T merge - keep as PeftModel for further LoRA training in Stage 2
        logger.info("Keeping LoRA adapter separate for Stage 2 training")
        
        # Load processor from base model (LoRA path typically doesn't have processor)
        processor = Qwen2VLProcessor.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
    else:
        # Load full model (non-LoRA)
        logger.info("Loading full model...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            stage1_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        processor = Qwen2VLProcessor.from_pretrained(
            stage1_model_path,
            trust_remote_code=True
        )
    
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Load dataset
    train_data = load_hf_dataset(config)
    dataset = prepare_dataset(train_data, processor)
    
    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=config["output_dir"],
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",

        num_train_epochs=config["training"]["num_epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=1,
        num_generations=4,
        max_prompt_length=400,
        max_completion_length=200,

        save_steps=1000,
        save_total_limit=1,
        logging_steps=20,
        warmup_steps=config["training"]["warmup_steps"],
        weight_decay=config["training"]["weight_decay"],
        max_grad_norm=config["training"]["max_grad_norm"],

        # save_strategy="no",
        bf16=True,
        temperature=0.7,
        top_p=0.9,
        report_to=config.get("report_to", "none")
    )
    
    # Initialize GRPO trainer with reward logging
    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=calculate_reward
    )
    
    # Reward logging callback for monitoring training
    class RewardLoggingCallback:
        def __init__(self):
            self.total_rewards = []
            
        def __call__(self, prompts, completions, **kwargs):
            rewards = calculate_reward(prompts, completions, **kwargs)
            
            # Track rewards
            if isinstance(rewards, list):
                self.total_rewards.extend(rewards)
            else:
                self.total_rewards.append(rewards)
            
            # Log progress every 50 steps
            if len(self.total_rewards) % 50 == 0:
                recent_avg = sum(self.total_rewards[-50:]) / 50
                logger.info(f"Step {len(self.total_rewards)}: Avg Reward = {recent_avg:.3f}")
                
                # Show sample prediction analysis
                last_completion = completions[-1] if isinstance(completions, list) and completions else str(completions)
                examples = kwargs.get('examples', [{}])
                last_example = examples[-1] if examples else {}
                
                # Quick analysis
                has_diagnosis = "<diagnosis>" in last_completion
                has_location = "<location>" in last_completion
                has_bbox = len(last_example.get('bbox', [])) == 4
                
                logger.info(f"  Format: Diag={'✓' if has_diagnosis else '✗'} | Loc={'✓' if has_location else '✗'} | BBox={'✓' if has_bbox else '✗'}")
                logger.info(f"  Sample: {last_completion[:80]}...")
            
            return rewards
    
    reward_logger = RewardLoggingCallback()
    trainer.reward_fn = reward_logger
    
    # Train
    logger.info("Starting GRPO training with reward logging...")
    trainer.train()
    
    # Save final model
    final_path = f"{config['output_dir']}/final"
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    
    logger.info(f"Training completed. Model saved to {final_path}")

if __name__ == "__main__":
    main()