# Facial Emotion Detection 😄😢😡

🎥 **[Watch Real-Time Demo](https://drive.google.com/file/d/1hbSXi4lOVQk7fP8GbSngoNVvNtmfB5zO/view)**

This project presents a robust deep learning pipeline that detects **five key facial emotions** in real-time using a webcam.  
It leverages **transfer learning** with **MobileNetV2**, enhanced by intermediate `Conv2D` blocks for domain-specific feature extraction.  
The model is trained on a **cleaned version of the FER2013 dataset** with data augmentation and regularization strategies.

---

## 💡 Motivation

The original [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) has limitations:
- Low-quality grayscale images  
- Imbalanced class distribution  
- Emotion classes that are visually confusing (e.g., *fear* vs *disgust*)

### ✅ Our Approach:
- Removed 2 confusing classes  
- Focused on: `neutral`, `happy`, `sad`, `angry`, `surprise`
- Applied **data augmentation**, **class weights**, and **early stopping**
- Used **MobileNetV2** as backbone and **cut into its intermediate layers**
- Added custom `Conv2D` blocks for better facial feature extraction
- Fine-tuned upper layers (top 30–40%) with a reduced learning rate

---

## 🔧 Key Features

- Transfer Learning from MobileNetV2 (ImageNet)
- Mid-network customization using Conv2D layers
- Data Augmentation (rotation, zoom, shift, shear, flip)
- EarlyStopping to prevent overfitting
- Class Weights to handle imbalanced classes
- Real-time emotion detection using OpenCV

---

## 📊 Dataset Summary

| Dataset   | Images   | Notes                         |
|-----------|----------|-------------------------------|
| FER2013   | ~30,000  | Grayscale resized to 224×224  |

Used only 5 classes: `neutral`, `happy`, `sad`, `angry`, `surprise`

---

## 📈 Performance

| Metric                | Value   |
|-----------------------|---------|
| Training Accuracy     | 85.6%   |
| Validation Accuracy   | 72.6%   |
| Test Accuracy         | 72.8%   |

---

## ▶️ Real-Time Inference

### 📦 Requirements

```bash
pip install tensorflow opencv-python numpy

---
### ▶️ How to Run
python realtime_emotion_detection.py

