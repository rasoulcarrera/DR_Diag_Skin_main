#!/usr/bin/env python3
"""
Skin Disease Diagnosis Model Training Script - Stage 1 (SFT)
Enhanced SFT for Vision-Language Models on ISIC Dataset with Spatial Awareness
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
    Qwen2VLForConditionalGeneration,
    AutoTokenizer, 
    AutoProcessor,
    get_linear_schedule_with_warmup,
    set_seed
)

from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import wandb

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOBBoxGenerator:
    """YOLO-based bbox generator for skin lesion detection"""
    
    def __init__(self, model_path=None, confidence_threshold=0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model"""
        try:
            from ultralytics import YOLO
            if self.model_path and os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                logger.info(f"Loaded custom YOLO model: {self.model_path}")
            else:
                # Use pre-trained YOLOv8 nano model
                self.model = YOLO('yolov8n.pt')
                logger.info("Loaded default YOLOv8n model for object detection")
        except ImportError:
            logger.warning("ultralytics not installed. Install with: pip install ultralytics")
            self.model = None
        except Exception as e:
            logger.warning(f"Could not load YOLO model: {e}")
            self.model = None
    
    def get_bbox(self, image_path):
        """Get bounding box from image using YOLO"""
        if not self.model:
            logger.warning("YOLO model not loaded, using fallback bbox")
            return self._fallback_bbox()
        
        try:
            results = self.model(image_path, verbose=False)
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    # Filter by confidence and get best detection
                    confidences = result.boxes.conf.cpu().numpy()
                    high_conf_indices = confidences >= self.confidence_threshold
                    
                    if any(high_conf_indices):
                        best_idx = confidences.argmax()
                        box = result.boxes.xyxy[best_idx].cpu().numpy()
                        return [int(x) for x in box]  # [x1, y1, x2, y2]
        except Exception as e:
            logger.warning(f"Error detecting bbox with YOLO: {e}")
        
        return self._fallback_bbox()
    
    def _fallback_bbox(self):
        """Return center bbox as fallback when YOLO detection fails"""
        return [112, 112, 336, 336]  # Center box for 224x224 image
    
    def get_model_info(self):
        """Get information about the loaded YOLO model"""
        return {
            "model_type": "yolo",
            "model_path": self.model_path or "yolov8n.pt",
            "confidence_threshold": self.confidence_threshold,
            "is_loaded": self.model is not None
        }

class SkinDiseaseDataset:
    """Dataset class with minimal text and optional bbox generation"""
    
    def __init__(self, image_dir: str, metadata_file: str, processor, tokenizer, max_length: int = 512,
                 image_size: int = 224, use_minimal_text: bool = True):
        self.image_dir = image_dir
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size
        self.use_minimal_text = use_minimal_text
        
        # Load metadata
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
        """Prepare dataset with minimal text and actual bbox coordinates"""
        self.data = []
        
        for item in self.metadata:
            image_path = os.path.join(self.image_dir, item['image_name'])
            if os.path.exists(image_path):
                
                if self.use_minimal_text:
                    # Minimal text approach
                    instruction = "Diagnose skin condition and locate lesion."
                    
                    # Use pre-computed bbox from metadata (no YOLO re-computation)
                    diagnosis = item.get('diagnosis', 'Unknown')
                    response = f"{diagnosis}"
                    
                    # Use bbox from metadata if available
                    if 'bbox' in item and item['bbox']:
                        bbox = item['bbox']
                        response += f" <bbox>{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}</bbox>"
                    else:
                        # Fallback to location info
                        location = item.get('location', item.get('anatom_site_general', 'center'))
                        response += f" <location>{location}</location>"
                    
                    # Add classification if available
                    if 'benign_malignant' in item:
                        response += f" <type>{item['benign_malignant']}</type>"
                
                else:
                    # Original verbose approach
                    instruction = """
                    Analyze this skin image carefully and provide:
                    1. Skin condition diagnosis
                    2. Precise location description (use spatial terms like 'upper left', 'center', 'lower right')
                    3. Approximate size estimation
                    4. Key visual features and concerning characteristics
                    5. Multiple lesions if present and their relative positions
                    """
                    
                    # Create detailed response with spatial information
                    response = f"Skin Condition: {item.get('diagnosis', 'Unknown')}\n"
                    
                    # Add location information
                    if 'location' in item:
                        response += f"Location: {item['location']}\n"
                    elif 'anatom_site_general' in item:
                        response += f"Body Location: {item['anatom_site_general']}\n"
                    else:
                        response += f"Location: Center of image\n"
                    
                    # Add size information
                    if 'size' in item:
                        response += f"Size: {item['size']}\n"
                    else:
                        response += f"Size: Approximately 1-2cm in diameter\n"
                    
                    # Add features
                    if 'features' in item:
                        response += f"Features: {item['features']}\n"
                    elif 'benign_malignant' in item:
                        response += f"Classification: {item['benign_malignant']}\n"
                        response += f"Features: This lesion shows characteristics typical of {item['benign_malignant']} growths\n"
                    else:
                        response += f"Features: Dark pigmentation, irregular borders, asymmetric shape\n"
                    
                    response += f"Description: This image shows a skin lesion that appears to be {item.get('diagnosis', 'a skin condition')} requiring medical attention."
                
                # Format the full text
                if self.use_minimal_text:
                    full_text = f"User: {instruction}\nAssistant: {response}"
                else:
                    full_text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
                
                self.data.append({
                    'image_path': image_path,
                    'full_text': full_text
                })
        
        text_type = "minimal" if self.use_minimal_text else "detailed"
        logger.info(f"Prepared {len(self.data)} samples with {text_type} text")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image for Qwen2VL processing
        try:
            from PIL import Image
            image = Image.open(item['image_path']).convert('RGB')
        except Exception as e:
            logger.warning(f"Error loading image {item['image_path']}: {e}")
            # Create a dummy image if loading fails
            image = Image.new('RGB', (self.image_size, self.image_size), color='white')
        
        # Prepare conversation format for Qwen2VL
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": item['full_text'].split('\n')[0]}  # Get instruction part
                ]
            },
            {
                "role": "assistant", 
                "content": [
                    {"type": "text", "text": item['full_text'].split('\n')[1] if '\n' in item['full_text'] else item['full_text']}  # Get response part
                ]
            }
        ]
        
        # Process with Qwen2VL processor
        try:
            text_input = self.processor.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=False
            )
        except:
            # Fallback to simple format
            text_input = f"<image>{item['full_text'].split('\n')[0]}"
        
        # Prepare examples in the format expected by processor
        examples = {
            "image": [image],
            "text": [text_input],
        }
        
        # Process using processor directly
        inputs = self.processor(
            images=[image],
            text=[text_input],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length
        )
        
        # Get processed tensors
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs.get("pixel_values", torch.zeros(1, 3, self.image_size, self.image_size)).squeeze(0)
        
        # Fix pixel_values shape if needed
        if len(pixel_values.shape) == 2:
            # Reshape flattened tensor to [3, H, W]
            total_pixels = pixel_values.shape[0] * pixel_values.shape[1]
            h = int((total_pixels // 3) ** 0.5)
            w = int(total_pixels // 3 // h)
            if h * w * 3 == total_pixels:
                pixel_values = pixel_values.reshape(3, h, w)
        
        # Process labels (target text)
        with self.tokenizer.as_target_tokenizer():
            label_encodings = self.tokenizer(
                [item['full_text'].split('\n')[1] if '\n' in item['full_text'] else item['full_text']],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length
            )
        labels = label_encodings["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask, 
            'labels': labels,
            'pixel_values': pixel_values
        }

class SkinDiseaseTrainer:
    """Enhanced trainer for skin disease model fine-tuning with spatial awareness"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        set_seed(config['training'].get('seed', 42))
        self.setup_model()
        self.setup_training_components()
        
        logger.info(f"Initialized trainer on device: {self.device}")
        logger.info(f"Training Stage: 1 (SFT with Enhanced Spatial Instructions)")
    
    def setup_model(self):
        """Setup Qwen2.5-VL model, tokenizer and processor"""
        model_name = self.config['model_name']
        
        # Load processor (handles both text and images for Qwen2VL)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load Qwen2VL model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.config['model'].get('use_fp16', True) else torch.float32,
            trust_remote_code=True,
            device_map='auto' if self.config['model'].get('use_device_map', True) else None
        )
        
        # Apply LoRA
        if self.config['lora'].get('use_lora', True):
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config['lora'].get('lora_r', 16),
                lora_alpha=self.config['lora'].get('lora_alpha', 32),
                lora_dropout=self.config['lora'].get('lora_dropout', 0.1),
                target_modules=self.config['lora'].get('lora_target_modules', ['q_proj', 'v_proj'])
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        logger.info(f"Loaded model: {model_name}")
    
    def setup_training_components(self):
        """Setup optimizer and scheduler"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training'].get('weight_decay', 0.01)
        )
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['training'].get('warmup_steps', 100),
            num_training_steps=self.config['training']['total_steps']
        )
        
        self.scaler = GradScaler() if self.config['model'].get('use_fp16', True) else None
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch in progress_bar:
            # Move batch to device for Qwen2VL
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            pixel_values = batch['pixel_values'].to(self.device)
            
            # Forward pass with Qwen2VL
            with autocast(enabled=self.config['model'].get('use_fp16', True)):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels
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
                    'stage': 'stage1_sft'
                })
        
        return total_loss / len(dataloader)
    
    def train(self, train_dataset, val_dataset=None):
        """Main training loop"""
        logger.info("Starting Stage 1 Training (SFT with Enhanced Spatial Instructions)...")
        logger.info("This stage focuses on disease identification and basic spatial awareness")
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['system'].get('num_workers', 4)
        )
        
        # Training loop
        for epoch in range(self.config['training']['num_epochs']):
            train_loss = self.train_epoch(train_dataloader, epoch)
            logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")
            
            # Log epoch metrics to wandb
            if wandb.run is not None:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'stage': 'stage1_sft'
                })
            
            # Save checkpoint
            if (epoch + 1) % self.config['training'].get('save_every', 2) == 0:
                self.save_model(f"stage1_checkpoint_epoch_{epoch}")
        
        logger.info("Stage 1 Training completed!")
        logger.info("Next: Use this model as base for Stage 2 (GRPO) for precise bounding boxes")
    
    def save_model(self, name):
        """Save model checkpoint"""
        save_path = os.path.join(self.config['output_dir'], name)
        os.makedirs(save_path, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save training info
        training_info = {
            'stage': 'stage1_sft',
            'model_name': self.config['model_name'],
            'training_config': self.config,
            'description': 'Stage 1 SFT model with enhanced spatial instructions for skin disease diagnosis'
        }
        
        with open(os.path.join(save_path, 'training_info.json'), 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"Stage 1 model saved to {save_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train skin disease diagnosis model - Stage 1 (SFT)')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Initialize wandb
    if config['system'].get('use_wandb', True):
        wandb.init(
            project=config['system'].get('wandb_project', 'skin-disease-diagnosis'),
            name=f"stage1-sft-{config['training']['num_epochs']}epochs",
            config=config,
            tags=['stage1', 'sft', 'spatial-awareness']
        )
    
    # Create trainer
    trainer = SkinDiseaseTrainer(config)
    
    # Load dataset
    train_dataset = SkinDiseaseDataset(
        image_dir=config['train_image_dir'],
        metadata_file=config['train_metadata_file'],
        processor=trainer.processor,
        tokenizer=trainer.tokenizer,
        max_length=config['model'].get('max_length', 512),
        image_size=config['model'].get('image_size', 224),
        use_minimal_text=config.get('use_minimal_text', True)
    )
    
    # Start training
    trainer.train(train_dataset)
    
    # Save final model
    trainer.save_model('stage1_final_model')
    
    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main() 