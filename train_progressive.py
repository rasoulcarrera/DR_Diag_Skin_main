#!/usr/bin/env python3
"""
Progressive Training Script - Final Solution
Stage 1: Basic bbox introduction
Stage 2: Precise bbox refinement with GRPO
"""

import os
import json
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_stage1():
    """Run Stage 1 SFT with basic bbox introduction"""
    logger.info("Starting Stage 1: Basic bbox introduction")
    
    # Update config for Stage 1
    config = {
        "model_name": "Qwen/Qwen2-VL-2B-Instruct",
        "training_stage": "stage1_sft",
        "output_dir": "./outputs_stage1",
        "train_image_dir": "./data/train/images",
        "train_metadata_file": "./data/train/metadata.json",
        
        "use_minimal_text": True,
        "use_bbox_model": True,
        
        "bbox_model_config": {
            "model_type": "yolo",
            "confidence_threshold": 0.5
        },
        
        "training": {
            "num_epochs": 10,
            "batch_size": 4,
            "learning_rate": 2e-5
        },
        
        "model": {
            "max_length": 512,
            "image_size": 224,
            "use_fp16": True
        },
        
        "lora": {
            "use_lora": True,
            "lora_r": 32,
            "lora_alpha": 64
        }
    }
    
    with open("config_stage1.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Run Stage 1
    cmd = ["python", "stage1_sft.py", "--config", "config_stage1.json"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("Stage 1 completed successfully")
        return "./outputs_stage1/stage1_final_model"
    else:
        logger.error(f"Stage 1 failed: {result.stderr}")
        return None

def run_stage2(stage1_model_path):
    """Run Stage 2 Enhanced GRPO for bbox refinement"""
    logger.info("Starting Stage 2: Enhanced GRPO bbox refinement")
    
    # Update config for Stage 2
    config = {
        "model_name": "Qwen/Qwen2-VL-2B-Instruct",
        "training_stage": "enhanced_grpo",
        "output_dir": "./outputs_stage2",
        "train_image_dir": "./data/train/images",
        "train_metadata_file": "./data/train/metadata_bbox.json",
        
        "bbox_model_config": {
            "model_type": "detr",  # Higher precision for Stage 2
            "confidence_threshold": 0.7
        },
        
        "chain_of_thought": {
            "enabled": True,
            "num_reasoning_steps": 4
        },
        
        "reward_config": {
            "bbox_weight": 0.6,
            "reasoning_weight": 0.25,
            "diagnosis_weight": 0.15,
            "iou_threshold": 0.7
        },
        
        "training": {
            "num_epochs": 8,
            "batch_size": 2,
            "learning_rate": 1e-5
        },
        
        "grpo_config": {
            "kl_coeff": 0.1,
            "temperature": 0.7
        }
    }
    
    with open("config_stage2.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Run Stage 2
    cmd = ["python", "enhanced_grpo.py", "--config", "config_stage2.json", "--stage1_model", stage1_model_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("Stage 2 completed successfully")
        return "./outputs_stage2/enhanced_grpo_final_model"
    else:
        logger.error(f"Stage 2 failed: {result.stderr}")
        return None

def main():
    """Main progressive training function"""
    logger.info("Starting Progressive Training Pipeline")
    logger.info("Stage 1: Basic bbox introduction with YOLO")
    logger.info("Stage 2: Precise bbox refinement with Enhanced GRPO")
    
    # Check prerequisites
    if not os.path.exists("./data/train/images"):
        logger.error("Training data not found. Please run prepare_data.py first")
        return
    
    # Stage 1: Basic bbox introduction
    stage1_model = run_stage1()
    if not stage1_model:
        logger.error("Stage 1 failed, stopping pipeline")
        return
    
    # Prepare Stage 2 data if needed
    if not os.path.exists("./data/train/metadata_bbox.json"):
        logger.info("Preparing Stage 2 bbox data...")
        cmd = ["python", "prepare_data.py", "--source_dir", "./isic_data", "--output_dir", "./data", "--stage", "2"]
        subprocess.run(cmd)
    
    # Stage 2: Enhanced GRPO refinement
    stage2_model = run_stage2(stage1_model)
    if not stage2_model:
        logger.error("Stage 2 failed")
        return
    
    logger.info("Progressive training completed successfully!")
    logger.info(f"Final model saved at: {stage2_model}")
    
    # Clean up temporary configs
    for config_file in ["config_stage1.json", "config_stage2.json"]:
        if os.path.exists(config_file):
            os.remove(config_file)
    
    print("\n" + "="*60)
    print("PROGRESSIVE TRAINING COMPLETED")
    print("="*60)
    print(f"Stage 1 Model: {stage1_model}")
    print(f"Stage 2 Model: {stage2_model}")
    print("\nModel Capabilities:")
    print("• Skin disease diagnosis")
    print("• Precise bounding box detection")
    print("• Chain-of-thought reasoning")
    print("• Minimal text responses")

if __name__ == "__main__":
    main()
