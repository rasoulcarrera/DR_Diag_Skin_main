# Skin Disease Diagnosis Model Training - 2-Stage Pipeline

This is a **2-stage training pipeline** for skin disease diagnosis using vision-language models:

- **Stage 1**: SFT with Enhanced Spatial Instructions (Current)
- **Stage 2**: GRPO for Precise Bounding Box Detection (Future)

## **Stage 1: Enhanced SFT Training (Current Focus)**

### **What It Does:**
✅ **Disease Identification**: Learn to identify skin conditions from images  
✅ **Spatial Awareness**: Provide precise location descriptions (text-based)  
✅ **Feature Analysis**: Identify concerning characteristics  
✅ **Multiple Lesion Detection**: Handle cases with multiple abnormalities  

### **Key Features:**
- **Enhanced Instructions**: Detailed prompts for better spatial understanding
- **Rich Metadata**: Location, size, features, concerning characteristics
- **LoRA Fine-tuning**: Efficient parameter updates
- **Wandb Logging**: Real-time training monitoring

## **Stage 2: GRPO Training (Future)**

### **What It Will Do:**
🎯 **Precise Bounding Boxes**: Output exact [x1, y1, x2, y2] coordinates  
🎯 **Spatial Precision**: Pixel-perfect localization  
🎯 **Advanced Localization**: Multiple lesion detection with individual boxes  

## **Setup**

### **1. Install Dependencies:**
```bash
pip install -r requirements.txt
```

### **2. Prepare Your Dataset:**

**For Stage 1 (Current):**
```bash
python prepare_data.py --source_dir /path/to/isic/dataset --output_dir ./data --stage 1
```

**For Stage 2 (Future):**
```bash
python prepare_data.py --source_dir /path/to/isic/dataset --output_dir ./data --stage 2
```

### **3. Update Configuration:**
Edit `config.json` for Stage 1 or `config_stage2.json` for Stage 2.

## **Training**

### **Stage 1 Training (Current):**
```bash
python stage1_sft.py --config config.json
```

### **Stage 2 Training (Future):**
```bash
python stage2_grpo.py --config config_stage2.json --stage1_model ./outputs/stage1_final_model
```

## **File Structure**

```
├── stage1_sft.py                    # Stage 1: SFT training
├── stage2_grpo.py                   # Stage 2: GRPO training  
├── prepare_data.py                  # Enhanced data preparation
├── config.json                      # Stage 1 configuration
├── config_stage2.json              # Stage 2 configuration
├── requirements.txt                 # Python dependencies
└── README.md                       # This file
```

## **Configuration**

### **Stage 1 Config (`config.json`):**
- `model_name`: Qwen2-VL-2B-Instruct
- `training_stage`: stage1_sft
- `num_epochs`: 15 (increased for better learning)
- `lora_r`: 32 (increased for better performance)

### **Stage 2 Config (`config_stage2.json`):**
- `training_stage`: stage2_grpo
- `batch_size`: 2 (smaller due to bounding box complexity)
- `learning_rate`: 1e-5 (lower for fine-tuning)
- `bbox_format`: [x1, y1, x2, y2]

## **Dataset Format**

### **Stage 1 (Enhanced SFT):**
```json
{
  "image_name": "melanoma_001.jpg",
  "diagnosis": "Melanoma",
  "location": "Upper left quadrant",
  "size": "1.5cm diameter",
  "features": "irregular_borders, asymmetric_shape",
  "concerning_features": ["asymmetry", "irregular_borders"]
}
```

### **Stage 2 (GRPO with Bounding Boxes):**
```json
{
  "image_name": "melanoma_001.jpg",
  "diagnosis": "Melanoma",
  "bbox": [150, 200, 300, 350],
  "confidence": 0.9
}
```

## **Training Progression**

### **Current (Stage 1):**
1. **Enhanced SFT Training** → Learn disease identification + basic spatial awareness
2. **Rich Text Output** → Detailed descriptions with location information
3. **Medical Knowledge** → Understand skin conditions and features

### **Future (Stage 2):**
1. **Use Stage 1 Model** → As base for GRPO training
2. **Add Bounding Boxes** → Manual annotation or object detection
3. **GRPO Training** → Learn precise coordinate output
4. **Final Result** → Disease identification + exact bounding boxes

## **What Was Enhanced**

✅ **Better Prompts**: More detailed spatial instructions  
✅ **Rich Metadata**: Location, size, features, concerning characteristics  
✅ **Stage Progression**: Clear path from SFT to GRPO  
✅ **Spatial Awareness**: Text-based localization before bounding boxes  
✅ **Medical Focus**: Enhanced medical terminology and features  

## **Benefits of This Approach**

1. **Progressive Learning**: Start simple, add complexity gradually
2. **Resource Efficient**: Stage 1 works on consumer GPUs
3. **Medical Accuracy**: Focus on disease identification first
4. **Spatial Understanding**: Build spatial awareness before precise coordinates
5. **Future Ready**: Easy transition to bounding box detection

## **Next Steps**

After completing Stage 1:
1. **Evaluate Performance** → Test disease identification accuracy
2. **Prepare Stage 2 Data** → Add bounding box annotations
3. **Run Stage 2 Training** → GRPO for precise localization
4. **Integration** → Combine both stages for complete solution

## **Notes**

- **Stage 1 is designed for experimentation** with minimal VRAM requirements
- **Stage 2 requires bounding box annotations** (manual or automated)
- **Progressive approach** ensures stable learning and better results
- **Medical applications** benefit from this staged learning approach
