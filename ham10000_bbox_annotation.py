import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import json

class HAM10000Annotator:
    def __init__(self, image_dir, csv_path, mask_dir, output_path):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.csv_path = csv_path
        self.output_path = output_path
        self.df = pd.read_csv(csv_path)
        
    def find_image_path(self, image_id):
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = self.image_dir / f"{image_id}{ext}"
            if img_path.exists():
                return img_path
        return None
    
    def find_mask_path(self, image_id):
        mask_path = self.mask_dir / f"{image_id}_segmentation.png"
        if mask_path.exists():
            return mask_path
        return None
    
    def extract_bbox_from_mask(self, mask_path):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
            
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
            
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        mask_area = cv2.contourArea(largest_contour)
        total_area = mask.shape[0] * mask.shape[1]
        coverage = mask_area / total_area
        
        center_x = x + w / 2
        center_y = y + h / 2
        img_width = mask.shape[1]
        img_height = mask.shape[0]
        
        return {
            'bbox': [float(x), float(y), float(x + w), float(y + h)],
            'area_coverage': float(coverage)
        }
    
    def process_dataset(self):
        annotations = []
        
        for _, row in self.df.iterrows():
            image_id = row['image_id']
            image_path = self.find_image_path(image_id)
            mask_path = self.find_mask_path(image_id)
            
            if image_path is None:
                continue
                
            if mask_path is not None:
                mask_data = self.extract_bbox_from_mask(mask_path)
                if mask_data is None:
                    continue
                    
                annotation = {
                    'lesion_id': row['lesion_id'],
                    'image_id': image_id,
                    'dx': row['dx'],
                    'dx_type': row['dx_type'],
                    'age': row['age'],
                    'sex': row['sex'],
                    'localization': row['localization'],
                    'bbox': mask_data['bbox'],
                    'area_coverage': mask_data['area_coverage']
                }
            else:
                annotation = {
                    'lesion_id': row['lesion_id'],
                    'image_id': image_id,
                    'dx': row['dx'],
                    'dx_type': row['dx_type'],
                    'age': row['age'],
                    'sex': row['sex'],
                    'localization': row['localization'],
                    'bbox': None,
                    'area_coverage': None
                }
                
            annotations.append(annotation)
        
        return annotations
    
    def save_annotations(self, annotations):
        with open(self.output_path, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        df_annotated = pd.DataFrame(annotations)
        csv_path = self.output_path.replace('.json', '.csv')
        df_annotated.to_csv(csv_path, index=False)

def main():
    image_dir = "D:/AI-ML/DATASET/HAM10000/HAM10000_images"
    mask_dir = "D:/AI-ML/DATASET/HAM10000_segmentations"
    csv_path = "D:/AI-ML/DATASET/HAM10000/HAM10000_metadata.csv"
    output_path = "D:/AI-ML/DATASET/HAM10000/ham10000_with_spatial_data.json"
    
    annotator = HAM10000Annotator(image_dir, csv_path, mask_dir, output_path)
    annotations = annotator.process_dataset()
    annotator.save_annotations(annotations)
    
    # masked_count = sum(1 for a in annotations if a['mask_available'])
    print(f"Processed {len(annotations)} images, with spatial data")

if __name__ == "__main__":
    main()
