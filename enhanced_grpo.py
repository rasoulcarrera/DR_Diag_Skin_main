#!/usr/bin/env python3
"""
Advanced reinforcement learning for skin disease bbox detection with reasoning chains
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
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import re

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

class EnhancedRewardModel:
    """EasyR1-inspired reward model for bbox accuracy and reasoning quality"""
    
    def __init__(self, config):
        self.config = config
        self.bbox_weight = config.get('bbox_weight', 0.6)
        self.reasoning_weight = config.get('reasoning_weight', 0.25)
        self.diagnosis_weight = config.get('diagnosis_weight', 0.15)
        self.iou_threshold = config.get('iou_threshold', 0.7)
        self.spatial_tolerance = config.get('spatial_tolerance', 10)
        
        # Reward scaling factors
        self.max_reward = config.get('max_reward', 1.0)
        self.min_reward = config.get('min_reward', -0.5)
    
    def compute_bbox_reward(self, pred_bbox: List[int], gt_bbox: List[int], 
                          image_size: Tuple[int, int] = (224, 224)) -> Dict[str, float]:
        """Enhanced bbox reward with multiple metrics"""
        iou = self._compute_iou(pred_bbox, gt_bbox)
        center_distance = self._compute_center_distance(pred_bbox, gt_bbox, image_size)
        size_similarity = self._compute_size_similarity(pred_bbox, gt_bbox)
        
        # Base IoU reward
        bbox_reward = iou
        
        # Center accuracy bonus (normalized by image diagonal)
        image_diagonal = np.sqrt(image_size[0]**2 + image_size[1]**2)
        center_reward = max(0, 1 - (center_distance / (image_diagonal * 0.25)))
        
        # Size accuracy bonus
        size_reward = size_similarity
        
        # Combined bbox reward
        total_bbox_reward = (
            0.6 * bbox_reward +      # IoU is most important
            0.25 * center_reward +   # Center accuracy
            0.15 * size_reward       # Size similarity
        )
        
        # Bonus for excellent localization
        if iou >= 0.8:
            total_bbox_reward += 0.2
        elif iou >= self.iou_threshold:
            total_bbox_reward += 0.1
        
        # Penalty for very poor localization
        if iou < 0.1:
            total_bbox_reward -= 0.3
        
        return {
            'iou': iou,
            'center_distance': center_distance,
            'size_similarity': size_similarity,
            'center_reward': center_reward,
            'size_reward': size_reward,
            'bbox_reward': max(0.0, min(1.0, total_bbox_reward))
        }
    
    def compute_reasoning_reward(self, reasoning_text: str, gt_diagnosis: str) -> Dict[str, float]:
        """Enhanced reasoning reward with step-by-step analysis"""
        reasoning_text = reasoning_text.lower()
        gt_diagnosis = gt_diagnosis.lower()
        
        # Chain-of-thought structure reward
        cot_reward = self._evaluate_cot_structure(reasoning_text)
        
        # Medical accuracy reward
        medical_reward = self._evaluate_medical_accuracy(reasoning_text, gt_diagnosis)
        
        # Reasoning depth reward
        depth_reward = self._evaluate_reasoning_depth(reasoning_text)
        
        # Language quality reward
        language_reward = self._evaluate_language_quality(reasoning_text)
        
        total_reasoning_reward = (
            0.4 * cot_reward +
            0.3 * medical_reward +
            0.2 * depth_reward +
            0.1 * language_reward
        )
        
        return {
            'cot_structure': cot_reward,
            'medical_accuracy': medical_reward,
            'reasoning_depth': depth_reward,
            'language_quality': language_reward,
            'reasoning_reward': min(1.0, total_reasoning_reward)
        }
    
    def compute_diagnosis_reward(self, pred_diagnosis: str, gt_diagnosis: str) -> float:
        """Reward for correct diagnosis"""
        pred_diagnosis = pred_diagnosis.lower().strip()
        gt_diagnosis = gt_diagnosis.lower().strip()
        
        # Exact match gets full reward
        if pred_diagnosis == gt_diagnosis:
            return 1.0
        
        # Partial match gets partial reward
        if pred_diagnosis in gt_diagnosis or gt_diagnosis in pred_diagnosis:
            return 0.7
        
        # Check for synonyms or related terms
        diagnosis_synonyms = {
            'melanoma': ['malignant melanoma', 'cutaneous melanoma'],
            'nevus': ['mole', 'nevi', 'melanocytic nevus'],
            'basal cell carcinoma': ['bcc', 'basal cell'],
            'seborrheic keratosis': ['sebkeratosis', 'seb ker'],
            'actinic keratosis': ['ak', 'solar keratosis']
        }
        
        for canonical, synonyms in diagnosis_synonyms.items():
            if (canonical in pred_diagnosis and gt_diagnosis in synonyms) or \
               (canonical in gt_diagnosis and pred_diagnosis in synonyms):
                return 0.8
        
        return 0.0
    
    def compute_total_reward(self, pred_bbox: List[int], gt_bbox: List[int],
                           reasoning_text: str, pred_diagnosis: str, gt_diagnosis: str,
                           image_size: Tuple[int, int] = (224, 224)) -> Dict[str, float]:
        """Compute comprehensive reward combining all factors"""
        
        bbox_metrics = self.compute_bbox_reward(pred_bbox, gt_bbox, image_size)
        reasoning_metrics = self.compute_reasoning_reward(reasoning_text, gt_diagnosis)
        diagnosis_reward = self.compute_diagnosis_reward(pred_diagnosis, gt_diagnosis)
        
        # Weighted combination
        total_reward = (
            self.bbox_weight * bbox_metrics['bbox_reward'] +
            self.reasoning_weight * reasoning_metrics['reasoning_reward'] +
            self.diagnosis_weight * diagnosis_reward
        )
        
        # Apply scaling
        total_reward = max(self.min_reward, min(self.max_reward, total_reward))
        
        # Combine all metrics
        all_metrics = {
            **bbox_metrics,
            **reasoning_metrics,
            'diagnosis_reward': diagnosis_reward,
            'total_reward': total_reward
        }
        
        return all_metrics
    
    def _compute_iou(self, box1: List[int], box2: List[int]) -> float:
        """Compute IoU of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_center_distance(self, box1: List[int], box2: List[int], 
                                image_size: Tuple[int, int]) -> float:
        """Compute distance between box centers"""
        cx1 = (box1[0] + box1[2]) / 2
        cy1 = (box1[1] + box1[3]) / 2
        cx2 = (box2[0] + box2[2]) / 2
        cy2 = (box2[1] + box2[3]) / 2
        
        return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    
    def _compute_size_similarity(self, box1: List[int], box2: List[int]) -> float:
        """Compute size similarity between boxes"""
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        if area1 == 0 and area2 == 0:
            return 1.0
        if area1 == 0 or area2 == 0:
            return 0.0
        
        ratio = min(area1, area2) / max(area1, area2)
        return ratio
    
    def _evaluate_cot_structure(self, text: str) -> float:
        """Evaluate chain-of-thought structure"""
        cot_indicators = [
            'first', 'then', 'next', 'finally', 'step', 'analyze',
            'examine', 'look at', 'observe', 'identify', 'conclude'
        ]
        
        score = 0.0
        for indicator in cot_indicators:
            if indicator in text:
                score += 0.1
        
        # Check for numbered steps
        if re.search(r'\d+\.', text):
            score += 0.3
        
        return min(1.0, score)
    
    def _evaluate_medical_accuracy(self, text: str, gt_diagnosis: str) -> float:
        """Evaluate medical terminology usage"""
        medical_terms = [
            'lesion', 'pigmentation', 'asymmetric', 'irregular', 'border',
            'malignant', 'benign', 'dermatoscopy', 'melanocytes', 'dermis',
            'epidermis', 'hyperkeratosis', 'dysplasia'
        ]
        
        score = 0.0
        for term in medical_terms:
            if term in text:
                score += 0.05
        
        # Bonus for correct diagnosis mention
        if gt_diagnosis in text:
            score += 0.3
        
        return min(1.0, score)
    
    def _evaluate_reasoning_depth(self, text: str) -> float:
        """Evaluate depth of reasoning"""
        depth_indicators = [
            'because', 'due to', 'indicates', 'suggests', 'shows',
            'characteristic', 'typical', 'evidence', 'features',
            'consistent with', 'appearance', 'pattern'
        ]
        
        score = 0.0
        for indicator in depth_indicators:
            if indicator in text:
                score += 0.1
        
        # Length bonus for detailed analysis
        word_count = len(text.split())
        if word_count > 20:
            score += 0.2
        elif word_count > 10:
            score += 0.1
        
        return min(1.0, score)
    
    def _evaluate_language_quality(self, text: str) -> float:
        """Evaluate language quality and coherence"""
        # Simple heuristics for language quality
        score = 0.5  # Base score
        
        # Penalty for very short responses
        if len(text.split()) < 5:
            score -= 0.3
        
        # Bonus for proper sentence structure
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        if sentence_count >= 2:
            score += 0.2
        
        # Penalty for repetitive text
        words = text.split()
        unique_ratio = len(set(words)) / len(words) if words else 0
        if unique_ratio < 0.5:
            score -= 0.2
        
        return max(0.0, min(1.0, score))

class ChainOfThoughtGenerator:
    """Generate sophisticated chain-of-thought prompts"""
    
    @staticmethod
    def generate_analysis_prompt(diagnosis: str = None) -> str:
        """Generate step-by-step analysis prompt"""
        return """Analyze this skin image systematically:

Step 1: Examine the overall appearance
- What is the general color and texture?
- Are there any obvious abnormalities?

Step 2: Assess key features
- Shape: Is it symmetric or asymmetric?
- Borders: Are they regular or irregular?
- Color: Is it uniform or varied?
- Size: What are the approximate dimensions?

Step 3: Identify the location
- Where exactly is the lesion located?
- Provide precise bounding box coordinates

Step 4: Make diagnosis
- Based on the features observed, what is the most likely diagnosis?
- What evidence supports this conclusion?

Now, let me analyze this image:"""
    
    @staticmethod
    def generate_localization_prompt() -> str:
        """Generate prompt for precise localization"""
        return """Look at this skin image and provide precise localization:

1. First, identify all visible lesions or abnormalities
2. For the primary lesion of concern:
   - Determine its exact position in the image
   - Estimate its boundaries carefully
   - Provide coordinates as [x1, y1, x2, y2] where:
     * (x1, y1) is the top-left corner
     * (x2, y2) is the bottom-right corner
     * Coordinates are in pixels

3. Explain your reasoning for the location choice

The lesion is located at:"""

class EnhancedGRPODataset:
    """Enhanced dataset with chain-of-thought prompts and reward computation"""
    
    def __init__(self, image_dir: str, metadata_file: str, tokenizer, config: Dict):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.get('max_length', 512)
        self.image_size = config.get('image_size', 224)
        self.use_cot = config.get('use_chain_of_thought', True)
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        # Initialize components
        self.reward_model = EnhancedRewardModel(config.get('reward_config', {}))
        self.cot_generator = ChainOfThoughtGenerator()
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare data with chain-of-thought prompts"""
        self.data = []
        
        for item in self.metadata:
            image_path = os.path.join(self.image_dir, item['image_name'])
            if not os.path.exists(image_path):
                continue
            
            if 'bbox' not in item:
                logger.warning(f"No bbox for {item['image_name']}, skipping...")
                continue
            
            # Generate prompts based on configuration
            if self.use_cot:
                instruction = self.cot_generator.generate_analysis_prompt(item.get('diagnosis'))
            else:
                instruction = f"Analyze this skin image and provide the diagnosis with exact bounding box coordinates for the {item.get('diagnosis', 'lesion')}."
            
            # Expected response format
            bbox = item['bbox']
            diagnosis = item.get('diagnosis', 'Unknown')
            
            # Create chain-of-thought response
            if self.use_cot:
                response = self._generate_cot_response(item, bbox, diagnosis)
            else:
                response = f"Diagnosis: {diagnosis}\nBounding box: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
            
            self.data.append({
                'image_path': image_path,
                'instruction': instruction,
                'response': response,
                'bbox': bbox,
                'diagnosis': diagnosis,
                'metadata': item,
                'full_text': f"Human: {instruction}\n\nAssistant: {response}"
            })
        
        logger.info(f"Prepared {len(self.data)} samples for Enhanced GRPO training")
    
    def _generate_cot_response(self, item: Dict, bbox: List[int], diagnosis: str) -> str:
        """Generate chain-of-thought response"""
        location = item.get('anatom_site_general', 'skin surface')
        classification = item.get('benign_malignant', 'unknown')
        
        response = f"""Step 1: Overall appearance
I can see a lesion on the {location} with distinct characteristics that require careful analysis.

Step 2: Key features assessment
- Shape: The lesion appears {'asymmetric' if classification == 'malignant' else 'symmetric'}
- Borders: {'Irregular' if classification == 'malignant' else 'Well-defined'} borders are visible
- Color: {'Varied pigmentation' if classification == 'malignant' else 'Uniform coloration'} is present
- Size: The lesion measures approximately {abs(bbox[2]-bbox[0])}x{abs(bbox[3]-bbox[1])} pixels

Step 3: Precise localization
The primary lesion is located at coordinates [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}].

Step 4: Diagnosis
Based on the observed features, this is consistent with {diagnosis}. The {'concerning' if classification == 'malignant' else 'benign'} characteristics {'warrant immediate medical attention' if classification == 'malignant' else 'suggest a benign condition'}."""
        
        return response
    
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
        
        # Tokenize
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
            'bbox': torch.tensor(item['bbox'], dtype=torch.float),
            'diagnosis': item['diagnosis'],
            'metadata': item['metadata']
        }

def main():
    """Main function for enhanced GRPO training"""
    parser = argparse.ArgumentParser(description='Enhanced GRPO Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--stage1_model', type=str, required=True, help='Path to Stage 1 model')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    logger.info("Enhanced GRPO training with EasyR1-style features")
    logger.info(f"Config: {args.config}")
    logger.info(f"Stage 1 model: {args.stage1_model}")
    
    # Create dataset
    dataset = EnhancedGRPODataset(
        image_dir=config['train_image_dir'],
        metadata_file=config['train_metadata_file'],
        tokenizer=None,  # Will be initialized with model
        config=config
    )
    
    logger.info(f"Dataset prepared with {len(dataset)} samples")
    logger.info("Enhanced GRPO implementation ready for training")

if __name__ == "__main__":
    main()
