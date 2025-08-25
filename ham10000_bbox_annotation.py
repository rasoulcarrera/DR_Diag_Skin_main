import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import json
from ultralytics import YOLO
import torch

class HAM10000Annotator:
    def __init__(self, image_dir, csv_path, output_path):
        self.image_dir = Path(image_dir)
        self.csv_path = csv_path
        self.output_path = output_path
        self.model = YOLO('yolov8n.pt')
        self.df = pd.read_csv(csv_path)
        
    def find_image_path(self, image_id):
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = self.image_dir / f"{image_id}{ext}"
            if img_path.exists():
                return img_path
        return None
    
    def detect_lesion(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            return None
            
        results = self.model(image, conf=0.3)
        
        if results[0].boxes is None or len(results[0].boxes) == 0:
            h, w = image.shape[:2]
            return [w*0.2, h*0.2, w*0.8, h*0.8, 0.5]
        
        best_box = results[0].boxes[0]
        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
        conf = best_box.conf[0].cpu().numpy()
        
        return [float(x1), float(y1), float(x2), float(y2), float(conf)]
    
    def process_dataset(self):
        annotations = []
        
        for _, row in self.df.iterrows():
            image_id = row['image_id']
            image_path = self.find_image_path(image_id)
            
            if image_path is None:
                continue
                
            bbox_data = self.detect_lesion(image_path)
            if bbox_data is None:
                continue
                
            annotation = {
                'lesion_id': row['lesion_id'],
                'image_id': image_id,
                'dx': row['dx'],
                'dx_type': row['dx_type'],
                'age': row['age'],
                'sex': row['sex'],
                'localization': row['localization'],
                'bbox': bbox_data[:4],
                'confidence': bbox_data[4]
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
    image_dir = "/kaggle/input/small-isic"
    csv_path = "/kaggle/input/small-isic/HAM10000_metadata.csv"
    output_path = "./ham10000_with_bbox.json"
    
    annotator = HAM10000Annotator(image_dir, csv_path, output_path)
    annotations = annotator.process_dataset()
    annotator.save_annotations(annotations)
    
    print(f"Processed {len(annotations)} images with bbox annotations")

if __name__ == "__main__":
    main()
