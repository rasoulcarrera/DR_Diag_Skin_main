---
license: cc-by-nc-4.0
task_categories:
- image-classification
- object-detection
tags:
- medical
- dermatology
- skin-disease
- bbox
- spatial-annotations
size_categories:
- 1K<n<10K
---

# HAM10000 with Spatial Annotations and Bounding Box Coordinates

Enhanced version of HAM10000 dataset with bounding box coordinates and spatial descriptions for skin lesion localization.

## Dataset Description

This dataset extends the original HAM10000 dermatology dataset with:
- Bounding box coordinates for lesion localization
- Spatial descriptions (e.g., "located in center-center region")
- Area coverage statistics
- Mask availability flags

## Features

- **image**: RGB skin lesion images
- **diagnosis**: Skin condition diagnosis codes (mel, nv, bkl, etc.)
- **bbox**: [x1, y1, x2, y2] bounding box coordinates
- **spatial_description**: Natural language location descriptions
- **area_coverage**: Lesion area relative to image size
- **localization**: Body part location
- **age/sex**: Patient demographics

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("abaryan/ham10000_bbox")
train_data = dataset["train"]

# Access image and annotations
sample = train_data[0]
image = sample["image"]
bbox = sample["bbox"]
description = sample["spatial_description"]
```

## Citation

Based on original HAM10000 dataset. Enhanced with spatial annotations for vision-language model training.
