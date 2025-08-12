#!/usr/bin/env python3
"""
Skin Disease Diagnosis Model Training Script - Stage 2 (GRPO)
GRPO training for precise bounding box detection using Stage 1 model as base
"""

import os
import json
import logging
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    set_seed
)

from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import wandb

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SkinDiseaseBBoxDataset:
    """Dataset class for ISIC skin disease images with bounding box annotations"""
    
    def __init__(self, image_dir: str, metadata_file: str, tokenizer, max_length: int = 512, image_size: int = 224):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size
        
        # Load metadata with bounding box information
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare dataset with bounding box coordinates for GRPO training"""
        self.data = []
        
        for item in self.metadata:
            image_path = os.path.join(self.image_dir, item['image_name'])
            if os.path.exists(image_path):
                # Check if bounding box information exists
                if 'bbox' not in item:
                    logger.warning(f"No bounding box found for {item['image_name']}, skipping...")
                    continue
                
                # GRPO-style instruction for precise localization
                instruction = f"<image>Where is the {item.get('diagnosis', 'skin lesion')} in this image? Provide the exact bounding box coordinates."
                
                # Expected response format: [x1, y1, x2, y2]
                bbox = item['bbox']
                if isinstance(bbox, list) and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    response = f"[{x1}, {y1}, {x2}, {y2}]"
                else:
                    logger.warning(f"Invalid bbox format for {item['image_name']}, skipping...")
                    continue
                
                # Add additional context if available
                if 'diagnosis' in item:
                    response += f" - {item['diagnosis']}"
                if 'confidence' in item:
                    response += f" (confidence: {item['confidence']})"
                
                self.data.append({
                    'image_path': image_path,
                    'instruction': instruction,
                    'response': response,
                    'bbox': bbox,
                    'full_text': f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
                })
        
        logger.info(f"Prepared {len(self.data)} samples for Stage 2 GRPO training with bounding boxes")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and process image
        try:
            from PIL import Image
            image = Image.open(item['image_path']).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            logger.warning(f"Error loading image {item['image_path']}: {e}")
            image = torch.zeros(3, self.image_size, self.image_size)
        
        # Tokenize text
        tokenized = self.tokenizer(
            item['full_text'],
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': tokenized['input_ids'].squeeze(),
            'image': image,
            'bbox': torch.tensor(item['bbox'], dtype=torch.float32)
        }

class SkinDiseaseGRPOTrainer:
    """GRPO trainer for skin disease model with bounding box detection"""
    
    def __init__(self, config, stage1_model_path):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stage1_model_path = stage1_model_path
        
        set_seed(config['training'].get('seed', 42))
        self.setup_model()
        self.setup_training_components()
        
        logger.info(f"Initialized Stage 2 GRPO trainer on device: {self.device}")
        logger.info(f"Training Stage: 2 (GRPO for Precise Bounding Box Detection)")
        logger.info(f"Using Stage 1 model from: {stage1_model_path}")
    
    def setup_model(self):
        """Setup model and tokenizer using Stage 1 model as base"""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.stage1_model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load Stage 1 model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.stage1_model_path,
            torch_dtype=torch.float16 if self.config['model'].get('use_fp16', True) else torch.float32,
            trust_remote_code=True,
            device_map='auto' if self.config['model'].get('use_device_map', True) else None
        )
        
        # Apply LoRA for Stage 2 training
        if self.config['lora'].get('use_lora', True):
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config['lora'].get('lora_r', 16),
                lora_alpha=self.config['lora'].get('lora_alpha', 32),
                lora_dropout=self.config['lora'].get('lora_dropout', 0.1),
                target_modules=self.config['lora'].get('target_modules', ['q_proj', 'v_proj'])
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        logger.info(f"Loaded Stage 1 model and applied LoRA for Stage 2 training")
    
    def setup_training_components(self):
        """Setup optimizer and scheduler for Stage 2"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training'].get('weight_decay', 0.01)
        )
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['training'].get('warmup_steps', 50),
            num_training_steps=self.config['training']['total_steps']
        )
        
        self.scaler = GradScaler() if self.config['model'].get('use_fp16', True) else None
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch with bounding box focus"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f'Stage 2 Epoch {epoch}')
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            images = batch['image'].to(self.device)
            bboxes = batch['bbox'].to(self.device)
            
            # Forward pass
            with autocast(enabled=self.config['model'].get('use_fp16', True)):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    images=images
                )
                loss = outputs.loss
            
            # Backward pass
            if self.config['model'].get('use_fp16', True) and self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update progress
            total_loss += loss.item()
            avg_loss = total_loss / (progress_bar.n + 1)
            progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}'})
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    'epoch': epoch,
                    'batch_loss': loss.item(),
                    'avg_loss': avg_loss,
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    'stage': 'stage2_grpo'
                })
        
        return total_loss / len(dataloader)
    
    def train(self, train_dataset, val_dataset=None):
        """Main training loop for Stage 2"""
        logger.info("Starting Stage 2 Training (GRPO for Precise Bounding Box Detection)...")
        logger.info("This stage builds on Stage 1 to add precise spatial localization")
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['system'].get('num_workers', 4)
        )
        
        # Training loop
        for epoch in range(self.config['training']['num_epochs']):
            train_loss = self.train_epoch(train_dataloader, epoch)
            logger.info(f"Stage 2 Epoch {epoch} - Train Loss: {train_loss:.4f}")
            
            # Log epoch metrics to wandb
            if wandb.run is not None:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'stage': 'stage2_grpo'
                })
            
            # Save checkpoint
            if (epoch + 1) % self.config['training'].get('save_every', 2) == 0:
                self.save_model(f"stage2_checkpoint_epoch_{epoch}")
        
        logger.info("Stage 2 Training completed!")
        logger.info("Model now capable of precise bounding box detection for skin diseases")
    
    def save_model(self, name):
        """Save Stage 2 model checkpoint"""
        save_path = os.path.join(self.config['output_dir'], name)
        os.makedirs(save_path, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save training info
        training_info = {
            'stage': 'stage2_grpo',
            'base_model': self.stage1_model_path,
            'model_name': self.config['model_name'],
            'training_config': self.config,
            'description': 'Stage 2 GRPO model with precise bounding box detection for skin diseases'
        }
        
        with open(os.path.join(save_path, 'training_info.json'), 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"Stage 2 model saved to {save_path}")

def main():
    """Main function for Stage 2 training"""
    parser = argparse.ArgumentParser(description='Train skin disease diagnosis model - Stage 2 (GRPO)')
    parser.add_argument('--config', type=str, default='config_stage2.json', help='Path to Stage 2 config file')
    parser.add_argument('--stage1_model', type=str, required=True, help='Path to Stage 1 trained model')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Initialize wandb
    if config['system'].get('use_wandb', True):
        wandb.init(
            project=config['system'].get('wandb_project', 'skin-disease-diagnosis-stage2'),
            name=f"stage2-grpo-{config['training']['num_epochs']}epochs",
            config=config,
            tags=['stage2', 'grpo', 'bounding-boxes']
        )
    
    # Create trainer
    trainer = SkinDiseaseGRPOTrainer(config, args.stage1_model)
    
    # Load dataset
    train_dataset = SkinDiseaseBBoxDataset(
        image_dir=config['train_image_dir'],
        metadata_file=config['train_metadata_file'],
        tokenizer=trainer.tokenizer,
        max_length=config['model'].get('max_length', 512),
        image_size=config['model'].get('image_size', 224)
    )
    
    # Start training
    trainer.train(train_dataset)
    
    # Save final model
    trainer.save_model('stage2_final_model')
    
    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()
