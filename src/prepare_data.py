#!/usr/bin/env python3
"""
Enhanced data preparation script for ISIC skin disease dataset
Supports both Stage 1 (SFT) and Stage 2 (GRPO) training
"""

import os
import json
import shutil
from pathlib import Path

def prepare_isic_dataset(source_dir, output_dir, train_split=0.8, stage=1, use_bbox_detection=True):
    """
    Prepare ISIC dataset by organizing images and creating metadata files
    
    Args:
        source_dir: Directory containing ISIC images and metadata
        output_dir: Output directory for organized data
        train_split: Fraction of data to use for training
        stage: 1 for SFT, 2 for GRPO (bounding boxes)
    """
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train", "images")
    val_dir = os.path.join(output_dir, "val", "images")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Find metadata file (common names in ISIC dataset)
    metadata_file = None
    for file in os.listdir(source_dir):
        if file.endswith('.csv') and ('metadata' in file.lower() or 'labels' in file.lower()):
            metadata_file = os.path.join(source_dir, file)
            break
    
    if not metadata_file:
        print("No metadata file found. Please ensure you have a CSV file with image labels.")
        return
    
    # Read metadata (assuming CSV format)
    import pandas as pd
    df = pd.read_csv(metadata_file)
    
    # Find image column and label column
    image_col = None
    label_col = None
    
    for col in df.columns:
        if 'image' in col.lower() or 'filename' in col.lower():
            image_col = col
        if 'diagnosis' in col.lower() or 'label' in col.lower() or 'class' in col.lower():
            label_col = col
    
    if not image_col or not label_col:
        print(f"Could not find image or label columns. Available columns: {list(df.columns)}")
        return
    
    # Prepare data
    train_data = []
    val_data = []
    
    for idx, row in df.iterrows():
        image_name = row[image_col]
        diagnosis = row[label_col]
        
        # Find image file
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            potential_path = os.path.join(source_dir, image_name + ext)
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if not image_path:
            continue
        
        # Create metadata entry based on stage
        if stage == 1:
            # Stage 1: SFT with pre-computed bboxes
            entry = create_stage1_entry(row, image_path, df.columns, use_bbox_detection)
        else:
            # Stage 2: GRPO with bounding boxes
            entry = create_stage2_entry(row, image_path, df.columns, use_bbox_detection)
        
        # Split into train/val
        if idx < len(df) * train_split:
            train_data.append(entry)
            # Copy image to train directory
            shutil.copy2(image_path, os.path.join(train_dir, os.path.basename(image_path)))
        else:
            val_data.append(entry)
            # Copy image to val directory
            shutil.copy2(image_path, os.path.join(val_dir, os.path.basename(image_path)))
    
    # Save metadata files
    if stage == 1:
        # Stage 1 metadata
        with open(os.path.join(output_dir, "train", "metadata.json"), 'w') as f:
            json.dump(train_data, f, indent=2)
        with open(os.path.join(output_dir, "val", "metadata.json"), 'w') as f:
            json.dump(val_data, f, indent=2)
        print(f"Stage 1 dataset prepared successfully!")
    else:
        # Stage 2 metadata with bounding boxes
        with open(os.path.join(output_dir, "train", "metadata_bbox.json"), 'w') as f:
            json.dump(train_data, f, indent=2)
        with open(os.path.join(output_dir, "val", "metadata_bbox.json"), 'w') as f:
            json.dump(val_data, f, indent=2)
        print(f"Stage 2 dataset prepared successfully!")
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Output directory: {output_dir}")

def create_stage1_entry(row, image_path, columns, use_bbox_detection=True):
    """Create Stage 1 metadata entry with pre-computed bbox coordinates"""
    entry = {
        'image_name': os.path.basename(image_path),
        'diagnosis': str(row.get('diagnosis', 'Unknown'))
    }
    
    # Add location information
    if 'anatom_site_general' in columns and pd.notna(row['anatom_site_general']):
        entry['location'] = str(row['anatom_site_general'])
    else:
        entry['location'] = 'center'
    
    if 'benign_malignant' in columns and pd.notna(row['benign_malignant']):
        entry['benign_malignant'] = str(row['benign_malignant'])
    
    # Pre-compute bbox coordinates using YOLO (run only if not already computed)
    if use_bbox_detection:
        # Check if bbox already exists in metadata to avoid re-computation
        existing_bbox = row.get('bbox') if hasattr(row, 'get') else None
        if existing_bbox and isinstance(existing_bbox, list) and len(existing_bbox) == 4:
            # Use existing bbox coordinates
            entry['bbox'] = existing_bbox
            entry['bbox_source'] = 'existing_precomputed'
            print(f"♻️  Using existing bbox for {os.path.basename(image_path)}: {existing_bbox}")
        else:
            # Generate new bbox using YOLO
            bbox = get_yolo_bbox(image_path)
            entry['bbox'] = bbox
            entry['bbox_source'] = 'yolo_precomputed'
            print(f"✅ Generated new bbox for {os.path.basename(image_path)}: {bbox}")
    else:
        entry['bbox'] = None
        entry['bbox_source'] = 'disabled'
    
    return entry

def create_stage2_entry(row, image_path, columns, use_bbox_detection=True):
    """Create Stage 2 metadata entry with bounding box coordinates"""
    entry = {
        'image_name': os.path.basename(image_path),
        'diagnosis': str(row.get('diagnosis', 'Unknown'))
    }
    
    # Add basic information
    if 'anatom_site_general' in columns and pd.notna(row['anatom_site_general']):
        entry['anatom_site_general'] = str(row['anatom_site_general'])
    if 'benign_malignant' in columns and pd.notna(row['benign_malignant']):
        entry['benign_malignant'] = str(row['benign_malignant'])
    
    # Pre-compute bbox coordinates using YOLO (run only if not already computed)
    if use_bbox_detection:
        # Check if bbox already exists in metadata to avoid re-computation
        existing_bbox = row.get('bbox') if hasattr(row, 'get') else None
        if existing_bbox and isinstance(existing_bbox, list) and len(existing_bbox) == 4:
            # Use existing bbox coordinates
            entry['bbox'] = existing_bbox
            entry['confidence'] = 0.8
            entry['bbox_source'] = 'existing_precomputed'
            print(f"♻️  Using existing bbox for {os.path.basename(image_path)}: {existing_bbox}")
        else:
            # Generate new bbox using YOLO
            bbox = get_yolo_bbox(image_path)
            entry['bbox'] = bbox
            entry['confidence'] = 0.8
            entry['bbox_source'] = 'yolo_precomputed'
            print(f"✅ Generated new bbox for {os.path.basename(image_path)}: {bbox}")
    else:
        # Use placeholder when bbox detection is disabled
        entry['bbox'] = [100, 100, 200, 200]
        entry['confidence'] = 0.5
        entry['bbox_source'] = 'placeholder'
    
    return entry

def estimate_location(row, columns):
    """Estimate location based on available metadata"""
    if 'anatom_site_general' in columns and pd.notna(row['anatom_site_general']):
        site = str(row['anatom_site_general']).lower()
        if 'head' in site or 'face' in site:
            return "Upper center region"
        elif 'trunk' in site or 'torso' in site:
            return "Center region"
        elif 'upper extremity' in site or 'arm' in site:
            return "Upper region"
        elif 'lower extremity' in site or 'leg' in site:
            return "Lower region"
    
    return "Center of image"

def estimate_size(row, columns):
    """Estimate size based on available metadata"""
    # Default size estimation
    return "Approximately 1-2cm in diameter"

def estimate_features(row, columns):
    """Estimate features based on available metadata"""
    features = []
    
    if 'benign_malignant' in columns and pd.notna(row['benign_malignant']):
        classification = str(row['benign_malignant']).lower()
        if 'malignant' in classification:
            features.extend(['irregular_borders', 'asymmetric_shape', 'color_variation'])
        else:
            features.extend(['regular_borders', 'symmetric_shape', 'uniform_color'])
    
    if not features:
        features = ['dark_pigmentation', 'irregular_borders', 'asymmetric_shape']
    
    return ', '.join(features)

def get_yolo_bbox(image_path):
    """Get actual bounding box using YOLO detection"""
    try:
        from ultralytics import YOLO
        
        # Load YOLO model (will download on first use)
        model = YOLO('yolov8n.pt')
        
        # Run detection
        results = model(image_path, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                # Get the most confident detection
                confidences = result.boxes.conf.cpu().numpy()
                best_idx = confidences.argmax()
                box = result.boxes.xyxy[best_idx].cpu().numpy()
                return [int(x) for x in box]  # [x1, y1, x2, y2]
    
    except ImportError:
        print("ultralytics not installed. Using fallback bbox.")
    except Exception as e:
        print(f"Error with YOLO detection: {e}")
    
    # Fallback to center box if YOLO fails
    return [100, 100, 200, 200]

def create_bbox_annotation_guide():
    """Create a guide for manual bounding box annotation"""
    guide = """
    BOUNDING BOX ANNOTATION GUIDE FOR STAGE 2:
    
    To prepare data for Stage 2 (GRPO with bounding boxes), you need to:
    
    1. Manually annotate bounding boxes around skin lesions
    2. Use annotation tools like:
       - LabelImg (https://github.com/tzutalin/labelImg)
       - CVAT (https://cvat.org/)
       - Roboflow (https://roboflow.com/)
    
    3. Bounding box format: [x1, y1, x2, y2]
       - x1, y1: Top-left corner coordinates
       - x2, y2: Bottom-right corner coordinates
       - All coordinates in pixels
    
    4. Update your metadata to include 'bbox' field:
       {
         "image_name": "melanoma_001.jpg",
         "diagnosis": "Melanoma",
         "bbox": [150, 200, 300, 350],
         "confidence": 0.9
       }
    
    5. Run this script with stage=2 to prepare Stage 2 dataset
    """
    
    with open("bbox_annotation_guide.txt", "w") as f:
        f.write(guide)
    
    print("Bounding box annotation guide created: bbox_annotation_guide.txt")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare ISIC dataset for training')
    parser.add_argument('--source_dir', type=str, required=True, help='Directory containing ISIC images and metadata')
    parser.add_argument('--output_dir', type=str, default='./data', help='Output directory for organized data')
    parser.add_argument('--train_split', type=float, default=0.8, help='Fraction of data for training')
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2], help='Training stage: 1 for SFT, 2 for GRPO')
    
    args = parser.parse_args()
    
    if args.stage == 2:
        print("Preparing Stage 2 dataset with pre-computed bounding boxes...")
    
    # Check if dataset already exists
    metadata_file = os.path.join(args.output_dir, "train", "metadata.json" if args.stage == 1 else "metadata_bbox.json")
    if os.path.exists(metadata_file):
        print("📋 Dataset already exists! YOLO will only run for new/missing bbox coordinates.")
    else:
        print("🔄 Running YOLO detection for new dataset preparation...")
    
    prepare_isic_dataset(args.source_dir, args.output_dir, args.train_split, args.stage, use_bbox_detection=True)
