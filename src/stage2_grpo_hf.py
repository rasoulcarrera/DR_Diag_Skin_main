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
    train_data = train_data.filter(lambda x: x.get('bbox') is not None and len(x.get('bbox', [])) == 4)
    
    logger.info(f"Loaded {len(train_data)} samples with spatial data")
    return train_data

def prepare_dataset(data, processor):
    """Prepare dataset for GRPO training"""
    def format_conversation(examples):
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
        
        # Handle both single example and batched examples
        if not isinstance(examples['diagnosis'], list):
            examples = {k: [v] for k, v in examples.items()}
        
        results = {
            "prompt": [],
            "images": [],
            "chosen": [],
            "completion": [],
            "answer": [],
            "bbox": [],
            "diagnosis": []
        }
        
        for i in range(len(examples['diagnosis'])):
            diagnosis_full = dx_names.get(examples['diagnosis'][i], examples['diagnosis'][i])
            prompt = "Analyze this skin lesion image. Respond in this exact format: <diagnosis>condition_name</diagnosis> <location>body_part</location> <bbox>[x1, y1, x2, y2]</bbox>"
            
            # Use localization field  
            body_location = examples['localization'][i]
            # Format bbox coordinates as integers
            bbox_coords = examples['bbox'][i]
            bbox_str = f"[{int(bbox_coords[0])}, {int(bbox_coords[1])}, {int(bbox_coords[2])}, {int(bbox_coords[3])}]"
            
            response = (
                f"<diagnosis>{diagnosis_full}</diagnosis> "
                f"<location>{body_location}</location> "
                f"<bbox>{bbox_str}</bbox>"
            )
            
            # Create conversation format - no system prompt, format instruction in user message
            conversation = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": response}]}
            ]
            
            # Build chat-templated prompt with generation prompt for proper assistant response
            prompt_text = processor.apply_chat_template(
                conversation[:-1],  # Only user message, not assistant
                tokenize=False,
                add_generation_prompt=True
            )
            
            results["prompt"].append(prompt_text)
            results["images"].append(examples["image"][i])
            results["chosen"].append(response)
            results["completion"].append(response)
            results["answer"].append(diagnosis_full)
            results["bbox"].append(examples["bbox"][i])
            results["diagnosis"].append(examples["diagnosis"][i])
            results["localization"] = examples["localization"]
        
        return results
    
    return data.map(format_conversation, num_proc=32, batched=True, batch_size=200)

def calculate_reward(prompts, completions, **kwargs):
    """Calculate reward for GRPO training with diagnosis and localization components."""
    import re
    
    # Extract completion text (GRPO passes list of strings)
    responses = completions
    
    # Get ground truth data from kwargs
    diagnosis = kwargs.get('diagnosis', [])
    localization = kwargs.get('localization', [])
    bbox = kwargs.get('bbox', [])
    
    # Ensure they're lists
    if not isinstance(diagnosis, list):
        diagnosis = [diagnosis] * len(responses)
    if not isinstance(localization, list):
        localization = [localization] * len(responses)
    if not isinstance(bbox, list):
        bbox = [bbox] * len(responses)
    
    # Reward weights
    weights = {"diagnosis": 0.6, "localization": 0.3, "bbox": 0.1}
    
    # Diagnosis mapping
    dx_names = {
        'akiec': 'actinic keratosis', 'bcc': 'basal cell carcinoma', 
        'bkl': 'benign keratosis-like lesion', 'df': 'dermatofibroma',
        'mel': 'melanoma', 'nv': 'melanocytic nevus', 'vasc': 'vascular lesion'
    }
    
    def extract_diagnosis(text):
        # First try XML tag
        match = re.search(r"<diagnosis>\s*(.*?)\s*</diagnosis>", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Fallback patterns for natural language
        match = re.search(r"(?:This appears to be|appears to be|diagnosis[:\s]*is|condition[:\s]*is)\s+([^.]+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Look for specific condition names
        conditions = ['melanoma', 'melanocytic nevus', 'basal cell carcinoma', 'actinic keratosis', 
                     'benign keratosis-like lesion', 'dermatofibroma', 'vascular lesion']
        for condition in conditions:
            if condition.lower() in text.lower():
                return condition
        
        return ""
    
    def extract_location(text):
        # First try XML tag
        match = re.search(r"<location>\s*(.*?)\s*</location>", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Fallback patterns for natural language
        match = re.search(r"(?:located in|location is|found on|situated on|on the)\s+(?:the\s+)?([^.]+)", text, re.IGNORECASE)
        if match:
            location = match.group(1).strip()
            # Clean up common phrases
            location = re.sub(r"(center-center region|region)", "", location).strip()
            return location
        
        # Look for body parts
        body_parts = ['face', 'trunk', 'back', 'chest', 'abdomen', 'upper extremity', 'lower extremity', 
                     'arm', 'leg', 'hand', 'foot', 'neck', 'scalp']
        for part in body_parts:
            if part.lower() in text.lower():
                return part
        
        return ""
    
    def extract_bbox(text):
        # Look for any coordinate pattern [x, y, x, y] in text
        match = re.search(r"\[(\d+),?\s*(\d+),?\s*(\d+),?\s*(\d+)\]", text)
        if match:
            try:
                return [int(match.group(i)) for i in range(1, 5)]
            except:
                pass
        return []
    
    rewards = []
    for i, response in enumerate(responses):
        reward = 0.0
        
        # Extract model predictions from response text
        pred_diag = extract_diagnosis(response).lower()
        pred_loc = extract_location(response).lower()
        pred_bbox = extract_bbox(response)
        
        # Get expected values from function arguments (like your MedMCQA)
        diagnosis_code = diagnosis[i] if i < len(diagnosis) else ''
        expected_diag = dx_names.get(diagnosis_code, diagnosis_code).lower()
        expected_loc = str(localization[i] if i < len(localization) else '').lower()
        sample_bbox = bbox[i] if i < len(bbox) else []
        
        # 1. Diagnosis accuracy (60%)
        if expected_diag and pred_diag:
            if expected_diag == pred_diag:
                reward += weights["diagnosis"]
            elif expected_diag in pred_diag or pred_diag in expected_diag:
                reward += weights["diagnosis"] * 0.8
        elif expected_diag:
            # Fallback: check if diagnosis appears anywhere in response (without tags)
            if expected_diag in response.lower():
                reward += weights["diagnosis"] * 0.4
        
        # 2. Localization accuracy (30%)
        if expected_loc and pred_loc:
            if expected_loc == pred_loc.strip():
                reward += weights["localization"]
            elif expected_loc in pred_loc:
                reward += weights["localization"] * 0.7
        
        # 3. Bbox prediction (10%)
        if isinstance(sample_bbox, (list, tuple)) and len(sample_bbox) == 4:
            if len(pred_bbox) == 4:
                # Simple reward for predicting bbox coordinates
                reward += weights["bbox"] * 0.8
            else:
                # Basic reward for bbox data existence
                reward += weights["bbox"] * 0.2
        
        # Penalties for poor formatting
        if not pred_diag: reward *= 0.8
        if not pred_loc: reward *= 0.9
        # Check if model includes bbox format (simple check)
        has_bbox_format = '<bbox>' in response.lower() and '</bbox>' in response.lower()
        if not has_bbox_format: reward *= 0.9
        if len(response.split()) > 120: reward *= 0.85
        
        # Debug output for first few examples
        if i < 3:
            # Simple bbox comparison
            bbox_found = "Yes" if len(pred_bbox) == 4 else "No"
            has_tags = "Yes" if '<bbox>' in response.lower() else "No"
            
            print(f"Example {i+1}: Diag='{pred_diag}' vs GT='{expected_diag}' | Loc='{pred_loc}' vs GT='{expected_loc}' | BBox={bbox_found} | Tags={has_tags} | Reward={reward:.2f}")
            if len(pred_bbox) == 4:
                print(f"BBox: {pred_bbox}")
            print(f"Model output: {response[:100]}...")
            if i == 0:  # Show prompt for first example
                prompt = prompts[0] if prompts else "No prompt available"
                print(f"PROMPT DEBUG (full): {prompt}")  # Show full prompt
        
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
            # device_map="auto",
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
            # device_map="auto",
            trust_remote_code=True
        )
        
        processor = Qwen2VLProcessor.from_pretrained(
            stage1_model_path,
            trust_remote_code=True
        )
    
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Fix for Qwen2-VL image token mismatch
    processor.tokenizer.padding_side = "left"
    
    # Force consistent image processing
    if hasattr(processor, 'image_processor'):
        processor.image_processor.do_resize = True
        processor.image_processor.size = {"height": 448, "width": 448}
        processor.image_processor.do_normalize = True
    
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
        num_generations=8,
        max_prompt_length=512,
        max_completion_length=200,

        save_steps=1000,
        save_total_limit=1,
        logging_steps=20,
        warmup_steps=config["training"]["warmup_steps"],
        weight_decay=config["training"]["weight_decay"],
        max_grad_norm=config["training"]["max_grad_norm"],

        bf16=True,
        temperature=0.7,
        top_p=0.9,
        report_to=config.get("report_to", "none"),
        
    )
    
    # Initialize GRPO trainer with reward logging
    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=[calculate_reward]
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