# Skin Disease Diagnosis with Spatial Awareness

Vision-language model for dermatology diagnosis with bbox detection and segmentation capabilities using Qwen2.5-VL and HAM10000 dataset.

## Training Pipeline

**Stage 1 (SFT)**: Foundation diagnosis training with basic spatial descriptions
**Stage 2 (GRPO)**: Reward-based optimization for precise bbox and segmentation

## Dataset Structure

```
ham10000_with_spatial_data.json
├── image_id: ISIC identifier
├── dx: diagnosis code (mel, nv, bkl, etc.)
├── bbox: [x1, y1, x2, y2] coordinates
├── spatial_description: "lesion located in center-center region"
└── mask_available: segmentation mask exists

## Training

### Stage 1 SFT
```bash
python src/stage1_sft.py --config src/config.json
```

### Stage 2 GRPO
```bash
python src/stage2_grpo.py --config src/config_stage2.json --stage1_model ./qwen2_5_vl_trained
or
accelerate launch --multi_gpu --num_processes=2 stage2_grpo_hf.py --config config_stage2_hf.json
```

## Data Preparation

### Extract spatial annotations from segmentation masks:
```bash
python ham10000_bbox_annotation.py
```

Outputs: `ham10000_with_spatial_data.json` and `ham10000_with_spatial_data.csv`

## Model Capabilities

- **Diagnosis**: 7 skin condition classes (melanoma, nevus, etc.)
- **Spatial awareness**: Natural language location descriptions  
- **Bbox detection**: Coordinate prediction for lesion localization
- **Segmentation**: IoU-based mask accuracy optimization

## Reward System (Stage 2)

- Diagnosis accuracy: 40% weight
- Spatial description: 30% weight  
- Segmentation IoU: 30% weight

## Requirements

- transformers>=4.35.0
- torch>=2.0.0
- Pillow, pandas, numpy
- opencv-python (for mask processing)

## Config Files

- `src/config.json`: Stage 1 training parameters
- `src/config_stage2.json`: Stage 2 GRPO settings with reward weights

## Expected Results

- Stage 1: 80-85% diagnosis accuracy, 100% spatial description accuracy
- Stage 2: Enhanced spatial precision with bbox coordinate prediction

## Citation

If you use this work, please cite:
```bibtex
@software{skin_disease_progressive_bbox,
  title={Progressive BBox Training for Skin Disease Diagnosis},
  author={Abaryan},
  year={2025}
}
```