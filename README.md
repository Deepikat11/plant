# 🌿 AI-Driven Crop Disease Detection Using Image Processing

> A hybrid deep learning system for real-time crop disease identification across crops like tomato, cotton, sugarcane, maize, rice, wheat, and potato using EfficientNetB0, ResNet-50V2, YOLOv4-tiny, and Random Forest.

---

##  Overview

Crop diseases significantly impact food security and agricultural economics. This project presents an advanced pipeline leveraging deep learning and ensemble methods to detect crop diseases from leaf images using image processing and AI.

---

##  Key Features

- **Leaf Region Detection:** YOLOv4-tiny for fast, lightweight leaf localization.
- **Deep Feature Extraction:** EfficientNetB0 (parameter-efficient) and ResNet-50V2 (residual learning) for robust feature analysis.
- **Multi-Crop Support:** Tomato, Cotton, Potato, Sugarcane, Maize, Rice, Wheat.
- **Preprocessing Pipeline:** Augmentation, normalization, resizing, and optional segmentation.
- **Hybrid Classification:** Random Forest as the final decision-maker for ensemble learning.
- **High Accuracy:** Boosted classification with feature fusion from dual CNNs.

---


## 🛠️ Tech Stack

| Component              | Technology                          |
|------------------------|--------------------------------------|
| Leaf Detection         | YOLOv8                               |
| Feature Extraction     | EfficientNetB0, ResNet-50V2          |
| Classification         | Random Forest (scikit-learn)         |
| Augmentation           | Albumentations, Keras, torchvision   |
| Preprocessing Tools    | OpenCV, PIL                          |
| Framework              | PyTorch, TensorFlow (optional ONNX) |

---

## 📊 Model Architecture

1. **YOLOv4-tiny**  
   → Localizes leaf or plant region from noisy backgrounds.
2. **EfficientNetB0 + ResNet-50V2**  
   → Extracts deep features from the cropped leaf image.
3. **Feature Fusion**  
   → Concatenates outputs into a single high-dimensional feature vector.
4. **Random Forest**  
   → Classifies disease class using fused features.
5. **(Optional)**  
   → Fully Connected + Softmax can be used as a neural classifier instead of RF.

---

## 🧪 Pipeline Steps

1. **Data Acquisition & Cleaning**
   - Download datasets, label images, clean duplicates.
   - Apply augmentations: Rotation, Flip, Shear, Zoom, etc.
2. **Image Preprocessing**
   - Resize: 224×224 (EfficientNet), 256×256 (ResNet), 416×416 (YOLOv4-tiny)
   - Normalize pixel values.
3. **Leaf Detection**
   - YOLOv4-tiny detects bounding box for the leaf region.
   - Crop image around the detected box.
4. **Feature Extraction**
   - EfficientNetB0 and ResNet-50V2 generate feature vectors.
5. **Feature Fusion**
   - Concatenate vectors into one (e.g., 3328-dim).
6. **Classification**
   - Trained Random Forest makes the final prediction.
7. **Evaluation**
   - Accuracy, Precision, Recall, F1-score, Confusion Matrix.

---
