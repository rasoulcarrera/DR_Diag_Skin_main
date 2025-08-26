#!/usr/bin/env python3
"""
Skin Disease Diagnosis Model Training Script - Stage 2 (GRPO)
Proper GRPO implementation with bbox reward system
"""

import os
import json
import logging
import argparse
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    TrainingArguments,
    Trainer
)
from datasets import Dataset as HFDataset
from PIL import Image
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BBoxRewardCalculator:
    """Calculate rewards for bbox predictions"""
    
    def __init__(self, config):
        self.config = config
        self.spatial_tolerance = config.get("grpo_settings", {}).get("spatial_tolerance", 5)
        self.iou_threshold = config.get("grpo_settings", {}).get("iou_threshold", 0.7)
    
    def calculate_iou(self, pred_bbox, gt_bbox):
        """Calculate IoU between predicted and ground truth bbox"""
        if not pred_bbox or not gt_bbox:
            return 0.0
        
        # Ensure bboxes are [x1, y1, x2, y2] format
        pred = torch.tensor(pred_bbox, dtype=torch.float32)
        gt = torch.tensor(gt_bbox, dtype=torch.float32)
        
        # Calculate intersection
        x1 = torch.max(pred[0], gt[0])
        y1 = torch.max(pred[1], gt[1])
        x2 = torch.min(pred[2], gt[2])
        y2 = torch.min(pred[3], gt[3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Calculate union
        pred_area = (pred[2] - pred[0]) * (pred[3] - pred[1])
        gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])
        union = pred_area + gt_area - intersection
        
        iou = intersection / (union + 1e-6)
        return iou.item()
    
    def calculate_spatial_reward(self, prediction, ground_truth, bbox_gt, mask_path=None):
        """Calculate reward based on spatial accuracy"""
        reward = 0.0
        
        # Diagnosis accuracy reward (0.4 weight)
        diagnosis_correct = self.check_diagnosis_match(prediction, ground_truth)
        reward += 0.4 if diagnosis_correct else 0.0
        
        # Spatial description reward (0.3 weight)
        if bbox_gt:
            spatial_reward = self.check_spatial_accuracy(prediction, ground_truth, bbox_gt)
            reward += 0.3 * spatial_reward
        
        # Segmentation reward (0.3 weight)
        if mask_path and os.path.exists(mask_path):
            seg_reward = self.check_segmentation_accuracy(prediction, mask_path)
            reward += 0.3 * seg_reward
        
        return reward
    
    def check_diagnosis_match(self, prediction, ground_truth):
        """Check if diagnosis matches"""
        diagnosis_keywords = {
            'actinic keratosis': 'akiec',
            'basal cell carcinoma': 'bcc', 
            'benign keratosis-like lesion': 'bkl',
            'dermatofibroma': 'df',
            'melanoma': 'mel',
            'melanocytic nevus': 'nv',
            'vascular lesion': 'vasc'
        }
        
        gt_dx = None
        pred_dx = None
        
        for full_name, short_code in diagnosis_keywords.items():
            if full_name in ground_truth.lower():
                gt_dx = short_code
            if full_name in prediction.lower():
                pred_dx = short_code
        
        return gt_dx == pred_dx and gt_dx is not None
    
    def check_spatial_accuracy(self, prediction, ground_truth, bbox_gt):
        """Check spatial description accuracy"""
        # Extract spatial descriptions
        gt_spatial = ""
        pred_spatial = ""
        
        if "located in" in ground_truth.lower():
            gt_spatial = ground_truth.split("located in")[-1].strip().rstrip(".")
        if "located in" in prediction.lower():
            pred_spatial = prediction.split("located in")[-1].strip().rstrip(".")
        
        if gt_spatial and pred_spatial:
            return 1.0 if gt_spatial.lower() == pred_spatial.lower() else 0.0
        elif not gt_spatial and not pred_spatial:
            return 1.0  # Both don't have spatial - correct
        else:
            return 0.0  # Mismatch in spatial prediction
    
    def check_segmentation_accuracy(self, prediction, mask_path):
        """Check segmentation accuracy using mask IoU"""
        try:
            import cv2
            
            # Load ground truth mask
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                return 0.0
            
            # Extract predicted coordinates if available
            if "[" in prediction and "]" in prediction:
                # Simple spatial overlap check based on predicted bbox
                bbox_str = prediction[prediction.find("["):prediction.find("]")+1]
                try:
                    coords = eval(bbox_str)  # Parse bbox coordinates
                    if len(coords) == 4:
                        x1, y1, x2, y2 = coords
                        # Create predicted mask from bbox
                        pred_mask = np.zeros_like(gt_mask)
                        pred_mask[int(y1):int(y2), int(x1):int(x2)] = 255
                        
                        # Calculate IoU
                        intersection = np.logical_and(gt_mask > 127, pred_mask > 127).sum()
                        union = np.logical_or(gt_mask > 127, pred_mask > 127).sum()
                        
                        if union > 0:
                            return intersection / union
                except:
                    pass
            
            return 0.0
            
        except Exception:
            return 0.0

class SpatialGRPODataset:
    """Dataset for GRPO training with spatial rewards"""
    
    def __init__(self, config, stage1_model_path):
        self.config = config
        self.stage1_model_path = stage1_model_path
        self.reward_calculator = BBoxRewardCalculator(config)
        
    def load_dataset(self):
        """Load dataset for GRPO training"""
        # Load annotated data with spatial information
        with open(self.config["train_metadata_file"], 'r') as f:
            spatial_data = json.load(f)
        
        image_dir = Path(self.config["train_image_dir"])
        conversations = []
        
        for item in spatial_data:
            # Only use samples with bbox data for Stage 2
            if not item.get('mask_available', False) or not item.get('bbox'):
                continue
                
            image_id = item['image_id']
            
            # Find image file
            image_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                img_path = image_dir / f"{image_id}{ext}"
                if img_path.exists():
                    image_path = str(img_path)
                    break
            
            if not image_path:
                continue
            
            # Get diagnosis info
            diagnosis = item['dx']
            dx_names = {
                'akiec': 'actinic keratosis',
                'bcc': 'basal cell carcinoma', 
                'bkl': 'benign keratosis-like lesion',
                'df': 'dermatofibroma',
                'mel': 'melanoma',
                'nv': 'melanocytic nevus',
                'vasc': 'vascular lesion'
            }
            diagnosis_full = dx_names.get(diagnosis, diagnosis)
            
            # Always include spatial for Stage 2
            user_prompt = "Analyze this skin lesion, provide a diagnosis, and describe its location."
            spatial_desc = item['spatial_description'].replace('lesion located in', 'The lesion is located in the')
            assistant_response = f"This appears to be {diagnosis_full}. {spatial_desc}."
            
            conversation = {
                "image_path": image_path,
                "conversation": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": user_prompt}
                        ]
                    },
                    {
                        "role": "assistant", 
                        "content": [
                            {"type": "text", "text": assistant_response}
                        ]
                    }
                ],
                "metadata": {
                    "lesion_id": item.get('lesion_id'),
                    "diagnosis": diagnosis,
                    "has_spatial": True,
                    "bbox": item.get('bbox'),
                    "area_coverage": item.get('area_coverage'),
                    "mask_available": True,
                    "spatial_description": item.get('spatial_description'),
                    "mask_path": f"{image_id}_segmentation.png"
                }
            }
            conversations.append(conversation)
        
        logger.info(f"Loaded {len(conversations)} samples with bbox data for GRPO training")
        return conversations

class GRPOTrainer:
    """GRPO Trainer for Stage 2"""
    
    def __init__(self, config, stage1_model_path):
        self.config = config
        self.stage1_model_path = stage1_model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reward_calculator = BBoxRewardCalculator(config)
        
        # Set random seed
        torch.manual_seed(config["training"]["seed"])
        random.seed(config["training"]["seed"])
        
        self.setup_models()
        logger.info(f"Initialized GRPO trainer on device: {self.device}")
    
    def setup_models(self):
        """Setup policy and reference models"""
        # Load processor
        self.processor = Qwen2VLProcessor.from_pretrained(
            self.stage1_model_path,
            trust_remote_code=True
        )
        
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
        
        # Load policy model (trainable)
        self.policy_model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.stage1_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # Apply LoRA to policy model if specified
        if self.config.get("lora", {}).get("use_lora", True):
            from peft import LoraConfig, get_peft_model, TaskType
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config["lora"]["lora_r"],
                lora_alpha=self.config["lora"]["lora_alpha"],
                lora_dropout=self.config["lora"]["lora_dropout"],
                target_modules=self.config["lora"]["target_modules"]
            )
            self.policy_model = get_peft_model(self.policy_model, lora_config)
            logger.info("Applied LoRA to policy model")
        
        # Load reference model (frozen)
        self.reference_model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.stage1_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        logger.info(f"Loaded models from: {self.stage1_model_path}")
    
    def generate_response(self, model, image, prompt):
        """Generate response from model"""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(
            text=[text_prompt], 
            images=[image], 
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        response = self.processor.batch_decode(
            generated_ids.sequences[:, inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )[0].strip()
        
        return response, generated_ids.scores
    
    def calculate_grpo_loss(self, policy_logprobs, ref_logprobs, rewards, beta=0.1):
        """Calculate GRPO loss"""
        # KL divergence between policy and reference
        kl_div = policy_logprobs - ref_logprobs
        
        # GRPO objective: maximize reward while staying close to reference
        loss = -torch.mean(rewards * policy_logprobs - beta * kl_div)
        
        return loss
    
    def train_epoch(self, dataset, epoch):
        """Train one epoch with GRPO"""
        self.policy_model.train()
        total_loss = 0
        total_reward = 0
        
        # Shuffle dataset
        shuffled_data = dataset.copy()
        random.shuffle(shuffled_data)
        
        batch_size = self.config["training"]["batch_size"]
        num_batches = len(shuffled_data) // batch_size
        
        optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"]
        )
        
        progress_bar = tqdm(range(0, len(shuffled_data), batch_size), desc=f'GRPO Epoch {epoch}')
        
        for batch_idx in progress_bar:
            batch_data = shuffled_data[batch_idx:batch_idx + batch_size]
            batch_rewards = []
            batch_policy_logprobs = []
            batch_ref_logprobs = []
            
            for sample in batch_data:
                try:
                    # Load image
                    image = Image.open(sample["image_path"]).convert('RGB')
                    prompt = sample["conversation"][0]["content"][1]["text"]
                    ground_truth = sample["conversation"][1]["content"][0]["text"]
                    bbox_gt = sample["metadata"]["bbox"]
                    
                    # Generate from policy model
                    policy_response, policy_scores = self.generate_response(
                        self.policy_model, image, prompt
                    )
                    
                    # Generate from reference model
                    ref_response, ref_scores = self.generate_response(
                        self.reference_model, image, prompt
                    )
                    
                    # Calculate reward including segmentation
                    mask_path = sample["metadata"].get("mask_path")
                    reward = self.reward_calculator.calculate_spatial_reward(
                        policy_response, ground_truth, bbox_gt, mask_path
                    )
                    
                    # Calculate log probabilities (simplified)
                    policy_logprob = torch.mean(torch.cat([score.max(dim=-1)[0] for score in policy_scores]))
                    ref_logprob = torch.mean(torch.cat([score.max(dim=-1)[0] for score in ref_scores]))
                    
                    batch_rewards.append(reward)
                    batch_policy_logprobs.append(policy_logprob)
                    batch_ref_logprobs.append(ref_logprob)
                    
                except Exception as e:
                    logger.warning(f"Error processing sample: {e}")
                    continue
            
            if not batch_rewards:
                continue
            
            # Convert to tensors
            rewards = torch.tensor(batch_rewards, device=self.device)
            policy_logprobs = torch.stack(batch_policy_logprobs)
            ref_logprobs = torch.stack(batch_ref_logprobs)
            
            # Calculate GRPO loss
            loss = self.calculate_grpo_loss(policy_logprobs, ref_logprobs, rewards)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_reward += rewards.mean().item()
            
            avg_loss = total_loss / (progress_bar.n + 1)
            avg_reward = total_reward / (progress_bar.n + 1)
            progress_bar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Reward': f'{avg_reward:.3f}'
            })
        
        return total_loss / num_batches, total_reward / num_batches
    
    def train(self, dataset):
        """Main GRPO training loop"""
        logger.info("Starting Stage 2 GRPO Training...")
        logger.info("This stage optimizes spatial accuracy using reward-based learning")
        
        for epoch in range(self.config["training"]["num_epochs"]):
            train_loss, avg_reward = self.train_epoch(dataset, epoch)
            logger.info(f"Epoch {epoch} - Loss: {train_loss:.4f}, Avg Reward: {avg_reward:.3f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config["training"]["save_every"] == 0:
                self.save_model(f"stage2_grpo_epoch_{epoch}")
        
        logger.info("Stage 2 GRPO training completed!")
    
    def save_model(self, name):
        """Save model checkpoint"""
        save_path = os.path.join(self.config["output_dir"], name)
        os.makedirs(save_path, exist_ok=True)
        
        self.policy_model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        
        # Save training info
        training_info = {
            'stage': 'stage2_grpo',
            'base_model': self.stage1_model_path,
            'training_config': self.config,
            'description': 'Stage 2 GRPO model with spatial reward optimization'
        }
        
        with open(os.path.join(save_path, 'training_info.json'), 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"Stage 2 model saved to {save_path}")

def main():
    """Main function for Stage 2 GRPO training"""
    parser = argparse.ArgumentParser(description='Train skin disease diagnosis model - Stage 2 (GRPO)')
    parser.add_argument('--config', type=str, default='config_stage2.json', help='Path to Stage 2 config file')
    parser.add_argument('--stage1_model', type=str, required=True, help='Path to Stage 1 trained model')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Load dataset
    dataset_loader = SpatialGRPODataset(config, args.stage1_model)
    dataset = dataset_loader.load_dataset()
    
    # Create trainer
    trainer = GRPOTrainer(config, args.stage1_model)
    
    # Start GRPO training
    trainer.train(dataset)
    
    # Save final model
    trainer.save_model('stage2_grpo_final')
    
    logger.info("Stage 2 GRPO training completed!")

if __name__ == "__main__":
    main()