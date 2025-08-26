#!/usr/bin/env python3
"""
Skin Disease Diagnosis Model Training Script - Stage 1 (SFT)
Based on working qwen2-5-vl-isic-training-reliable.ipynb approach
"""

import os
import json
import logging
import argparse
import random
from pathlib import Path

import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    TrainingArguments,
    Trainer
)
from datasets import Dataset as HFDataset
from PIL import Image
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpatialDatasetLoader:
    """Load spatial dataset matching notebook approach"""
    
    def __init__(self, config):
        self.config = config
        
    def load_spatial_dataset(self):
        """Load spatial dataset exactly like notebook"""
        # Load annotated data with spatial information
        with open(self.config["spatial_dataset_file"], 'r') as f:
            spatial_data = json.load(f)
        
        image_dir = Path(self.config["train_image_dir"])
        conversations = []
        
        for item in spatial_data:
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
            
            # Create conversation with optional spatial awareness
            use_spatial = (self.config["include_spatial_descriptions"] and 
                          item.get('mask_available', False) and 
                          random.random() < self.config["spatial_description_ratio"])
            
            if use_spatial and item.get('spatial_description'):
                user_prompt = "Analyze this skin lesion, provide a diagnosis, and describe its location."
                spatial_desc = item['spatial_description'].replace('lesion located in', 'The lesion is located in the')
                assistant_response = f"This appears to be {diagnosis_full}. {spatial_desc}."
            else:
                user_prompt = "Analyze this skin lesion and provide a diagnosis."
                assistant_response = f"This appears to be {diagnosis_full}."
            
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
                    "has_spatial": use_spatial,
                    "bbox": item.get('bbox'),
                    "area_coverage": item.get('area_coverage'),
                    "mask_available": item.get('mask_available', False)
                }
            }
            conversations.append(conversation)
        
        return conversations

def collate_fn(examples, processor, config):
    """Enhanced data collator for spatial-aware dataset"""
    texts = []
    images = []
    
    for example in examples:
        image = Image.open(example["image_path"]).convert('RGB')
        images.append(image)
        
        text = processor.apply_chat_template(
            example["conversation"], 
            tokenize=False, 
            add_generation_prompt=False
        )
        texts.append(text)
    
    batch = processor(
        text=texts,
        images=images, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config["max_length"]
    )
    
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    # Mask image tokens in labels to prevent learning image embeddings as text
    if hasattr(processor, 'image_token_id'):
        labels[labels == processor.image_token_id] = -100
    
    batch["labels"] = labels
    return batch

class SkinDiseaseTrainer:
    """Stage 1 SFT Trainer matching notebook approach"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seed
        torch.manual_seed(config["seed"])
        random.seed(config["seed"])
        
        self.setup_model()
        logger.info(f"Initialized Stage 1 SFT trainer on device: {self.device}")
    
    def setup_model(self):
        """Setup model and processor exactly like notebook"""
        model_name = self.config["model_name"]
        
        # Load model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # Load processor
        self.processor = Qwen2VLProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Setup pad token
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
        
        logger.info(f"Model loaded: {model_name}")
    
    def prepare_datasets(self, dataset):
        """Prepare train/eval split like notebook"""
        # Shuffle dataset to avoid bias
        shuffled_dataset = dataset.copy()
        random.shuffle(shuffled_dataset)
        
        # Use 70/30 split for better evaluation
        train_size = int(0.7 * len(shuffled_dataset))
        train_data = shuffled_dataset[:train_size]
        eval_data = shuffled_dataset[train_size:]
        
        train_dataset = HFDataset.from_list(train_data)
        eval_dataset = HFDataset.from_list(eval_data)
        
        logger.info(f"Training: {len(train_data)} | Evaluation: {len(eval_data)}")
        return train_dataset, eval_dataset
    
    def train(self, dataset):
        """Main training loop"""
        logger.info("Starting Stage 1 SFT Training...")
        
        # Prepare datasets
        train_dataset, eval_dataset = self.prepare_datasets(dataset)
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config["output_dir"],
            per_device_train_batch_size=self.config["per_device_train_batch_size"],
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
            num_train_epochs=self.config["num_train_epochs"],
            learning_rate=self.config["learning_rate"],
            
            # Learning rate scheduling and regularization
            warmup_steps=self.config["warmup_steps"],
            lr_scheduler_type="cosine",
            weight_decay=self.config["weight_decay"],
            max_grad_norm=self.config["max_grad_norm"],
            
            # Logging and saving
            logging_steps=100,
            save_steps=1000,
            eval_steps=1000,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_total_limit=2,
            
            # Memory and performance optimizations
            remove_unused_columns=False,
            bf16=True,
            report_to="none"
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=lambda x: collate_fn(x, self.processor, self.config),
            tokenizer=self.processor.tokenizer,
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training complete!")
        
        # Save model and processor
        trainer.save_model(self.config["output_dir"])
        self.processor.save_pretrained(self.config["output_dir"])
        logger.info(f"Model saved to {self.config['output_dir']}")
        
        return trainer

def test_model_inference(model, processor, test_samples, config, num_tests=10):
    """Test model inference like notebook"""
    results = []
    
    for i, sample in enumerate(test_samples[:num_tests]):
        print(f"\n--- Test {i+1}/{num_tests} ---")
        
        try:
            # Load test image
            image_path = sample["image_path"]
            test_image = Image.open(image_path).convert('RGB')
            print(f"Image: {os.path.basename(image_path)}")
            
            # Create conversation for inference
            has_spatial_data = sample["metadata"]["has_spatial"]
            if has_spatial_data:
                user_prompt = "Analyze this skin lesion, provide a diagnosis, and describe its location."
            else:
                user_prompt = "Analyze this skin lesion and provide a diagnosis."
                
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_prompt}
                    ]
                }
            ]
            
            # Process input
            text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(
                text=[text_prompt], 
                images=[test_image], 
                return_tensors="pt"
            ).to(model.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            
            # Decode response
            response = processor.batch_decode(
                generated_ids[:, inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )[0].strip()
            
            # Get ground truth
            ground_truth = sample["conversation"][1]["content"][0]["text"]
            
            print(f"Ground Truth: {ground_truth}")
            print(f"Model Output: {response}")
            
            results.append({
                "image": os.path.basename(image_path),
                "ground_truth": ground_truth,
                "prediction": response,
                "success": True
            })
            
        except Exception as e:
            print(f"❌ Error processing test {i+1}: {e}")
            results.append({
                "image": f"test_{i+1}",
                "error": str(e),
                "success": False
            })
    
    # Summary
    successful_tests = sum(1 for r in results if r["success"])
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY: {successful_tests}/{len(results)} tests passed")
    print(f"{'='*50}")
    
    return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train skin disease diagnosis model - Stage 1 (SFT)')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Load dataset
    loader = SpatialDatasetLoader(config)
    dataset = loader.load_spatial_dataset()
    
    spatial_count = sum(1 for c in dataset if c['metadata']['has_spatial'])
    logger.info(f"Dataset loaded: {len(dataset)} samples, {spatial_count} with spatial descriptions")
    
    # Create trainer
    trainer = SkinDiseaseTrainer(config)
    
    # Start training
    trained_model = trainer.train(dataset)
    
    # Test the model
    logger.info("Testing trained model...")
    test_results = test_model_inference(
        trained_model.model, 
        trainer.processor, 
        dataset, 
        config, 
        num_tests=10
    )
    
    logger.info("Stage 1 SFT training completed!")

if __name__ == "__main__":
    main()