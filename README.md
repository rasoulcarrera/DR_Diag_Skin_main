# Skin Disease Diagnosis with Progressive BBox Training

An advanced AI system for medical skin disease diagnosis that combines disease identification, precise localization, and medical reasoning in a two-stage progressive training approach.

## Model Purpose and Goals

This system is designed to assist medical professionals in skin disease diagnosis by providing:

1. **Disease Identification**: Accurate classification of skin conditions (melanoma, nevus, basal cell carcinoma, etc.)
2. **Precise Localization**: Exact bounding box coordinates to pinpoint lesion locations
3. **Medical Reasoning**: Chain-of-thought explanations for diagnostic decisions
4. **Spatial Awareness**: Understanding of lesion position, size, and characteristics

The model learns progressively:
- **Stage 1 [SFT]**: Establishes foundation for disease recognition and basic spatial understanding
- **Stage 2 [GRPO]**: Refines localization precision and develops detailed medical reasoning capabilities

## Overview

This system implements a progressive training approach where the model first learns basic concepts and then develops advanced diagnostic and reasoning skills through reinforcement learning.

## Features

- 🎯 **Progressive BBox Training**: Basic introduction → Precise refinement
- 🤖 **YOLO BBox Detection**: Automated lesion localization
- 💬 **Minimal Text Responses**: Concise medical diagnosis format
- 🧠 **Chain-of-Thought**: Medical reasoning for Stage 2
- ⚙️ **Configurable**: Easy enable/disable of bbox models

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/rasoulcarrera/DR_Diag_Skin_main.git
cd DR_Diag_Skin_main

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Prepare your ISIC dataset
python prepare_data.py --source_dir ./isic_data --output_dir ./data --stage 1
```

### 3. Progressive Training

**Option A: Automatic Progressive Training**
```bash
python train_progressive.py
```

**Option B: Manual Stage-by-Stage**
```bash
# Stage 1: Basic bbox introduction
python stage1_sft.py --config config.json

# Stage 2: Enhanced GRPO refinement
python enhanced_grpo.py --config config_stage2.json --stage1_model ./outputs/stage1_final_model
```

## Configuration

### YOLO BBox Configuration

Enable/disable YOLO bounding box detection:

```json
{
  "use_bbox_model": true,
  "bbox_model_config": {
    "model_path": null,             // Use default YOLOv8n or specify custom model
    "confidence_threshold": 0.5     // Adjust detection sensitivity
  }
}
```

### Minimal Text Format

**Stage 1 Output:**
```
Melanoma <bbox>150,200,300,350</bbox> <type>malignant</type>
```

**Stage 2 Output:**
```
Step 1: Overall appearance analysis...
Step 2: Feature assessment...
Step 3: Precise localization at [142,186,298,342]
Step 4: Diagnosis: Melanoma with detailed reasoning
```

## How It Works

The system uses a sophisticated approach to medical image analysis:

1. **Vision-Language Integration**: Combines advanced computer vision with natural language processing
2. **Progressive Learning**: Starts with basic pattern recognition and evolves to complex medical reasoning
3. **YOLO Detection**: Uses YOLOv8 for accurate and fast bounding box detection
4. **Reward-Based Refinement**: Uses reinforcement learning to improve diagnostic accuracy and reasoning quality

## Training Stages

### Stage 1: Basic BBox Introduction
- **Duration**: 2-4 hours
- **BBox Model**: YOLO (fast, good baseline)
- **Text**: Minimal responses
- **Goal**: Learn basic spatial awareness

### Stage 2: Enhanced GRPO Refinement
- **Duration**: 4-6 hours  
- **BBox Model**: DETR (higher precision)
- **Text**: Chain-of-thought reasoning
- **Goal**: Precise localization + medical reasoning

## File Structure

```
DR_Diag_Skin/
├── config.json                 # Main configuration
├── config_stage2.json          # Stage 2 GRPO config
├── stage1_sft.py              # Stage 1 training
├── enhanced_grpo.py           # Stage 2 enhanced GRPO
├── prepare_data.py            # Data preparation
├── train_progressive.py       # Automatic progressive training
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Usage Examples

### Quick Prototyping (Stage 1 Only)
```bash
# Disable Stage 2, use YOLO for fast results
python stage1_sft.py --config config.json
```

### Medical Production (Both Stages)
```bash
# Full progressive training for highest accuracy
python train_progressive.py
```

### Research Setup (Custom BBox Model)
```bash
# Edit config.json to use GroundingDINO
# Then run progressive training
python train_progressive.py
```

## Advanced Configuration

### YOLO Detection Model

The system uses YOLOv8 for bounding box detection:

- **Fast Performance**: Optimized for real-time inference
- **Good Accuracy**: Reliable detection for medical images
- **Easy Setup**: Automatic model download on first use
- **Configurable**: Adjustable confidence thresholds

YOLO provides the optimal balance of speed and accuracy for medical image analysis.

### Disable BBox Detection
```json
{
  "use_bbox_model": false
}
```

## Dependencies

- `torch>=2.0.0`
- `transformers>=4.35.0`
- `peft>=0.6.0`
- `ultralytics>=8.0.0` (for YOLO)
- `timm>=0.9.0` (for DETR)

## Troubleshooting

**CUDA Out of Memory:**
- Reduce `batch_size` in config
- Use `gradient_accumulation_steps`

**BBox Model Issues:**
- YOLO: `pip install ultralytics`
- DETR: Update transformers
- GroundingDINO: Follow official setup

**Training Issues:**
- Check data paths in config
- Verify ISIC dataset format
- Monitor GPU memory usage

## Model Output Examples

The trained model produces medical-grade diagnostic outputs with increasing sophistication:

**Stage 1 Output (Foundation):**
```
User: Diagnose skin condition and locate lesion.
Assistant: Melanoma <bbox>150,200,300,350</bbox> <type>malignant</type>
```

**Stage 2 Output (Advanced Reasoning):**
```
User: Analyze this skin image systematically.
Assistant: 
Step 1: Overall appearance - I observe a dark, irregularly shaped lesion with varied pigmentation
Step 2: Feature assessment - The borders appear asymmetric with color variation typical of concerning lesions
Step 3: Precise localization - The primary lesion is located at coordinates [142,186,298,342]
Step 4: Medical diagnosis - This appears consistent with melanoma. The irregular borders, asymmetric shape, and color variation warrant immediate medical attention for biopsy confirmation.
```

This progressive approach ensures the model develops from basic recognition to sophisticated medical reasoning capabilities.

## License

This project is licensed under the MIT License.

## Citation

If you use this work, please cite:
```bibtex
@software{skin_disease_progressive_bbox,
  title={Progressive BBox Training for Skin Disease Diagnosis},
  author={Abaryan},
  year={2025}
}
```