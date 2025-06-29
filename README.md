# Facial Emotion Detection

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

🎥 **[Watch Real-Time Demo](https://drive.google.com/file/d/1D02W118gHPeF8MFtjWGWzH7O_uNRSkQ6/view)**

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
```
---
### ▶️ How to Run

```bash
python realtime_emotion_detection.py
```

The webcam will open and display live predictions for one of the following emotions:
neutral, happy, sad, angry, surprise

---
## 🔗 Model Download

📦 You can download the trained model here:  
➡️ [Download emotion_model.h5](https://drive.google.com/uc?id=1PYUCvC7e1VviaAInCyobcQntajFLTT18&export=download)

Once downloaded, place the file in the root directory of the project (same level as `realtime_emotion_detection.py`) before running the script:

---

## 📂 Files in the Repository

| File                             | Description                                 |
|----------------------------------|---------------------------------------------|
| `train_emotion_model.ipynb`      | Training notebook using FER2013             |
| `realtime_emotion_detection.py`  | Script for real-time webcam detection       |
| `realtime_emotion_detection.ipynb` | Jupyter version for real-time testing     |
| `emotion_labels.npy`             | Saved class labels                          |
| `emotion_model.h5` *(external)*  | Trained model weights (download separately) |
| `requirements.txt`               | Python dependencies                         |
| `README.md`                      | Project documentation                       |

---

## 👤 Author

**Youssef Osama Fawzy**  
[GitHub](https://github.com/Youssef-Osama1) • [LinkedIn](https://www.linkedin.com/in/youssef-osama-770a19297/)
