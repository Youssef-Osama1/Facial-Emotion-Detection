# Facial Emotion Detection 😄😢😡

🎥 [Watch Real-Time Demo](https://drive.google.com/file/d/1hbSXi4lOVQk7fP8GbSngoNVvNtmfB5zO/view)

This project presents a robust deep learning pipeline that detects **five key facial emotions** in real-time using a webcam.  
It leverages **transfer learning via MobileNetV2**, with a unique fine-tuning stage using **personally captured images** to boost recognition accuracy on the target face.

---

## 💡 Motivation

The original [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) suffers from:

- Low-resolution grayscale images  
- Class imbalance  
- Emotion classes that are visually confusing (e.g., *fear* vs *disgust*)

### ✅ Our Strategy:

- Trained initial model on FER2013 using MobileNetV2 (with intermediate Conv2D blocks)
- Removed 2 confusing classes → focused on: `neutral`, `happy`, `sad`, `angry`, `surprise`
- Applied **data augmentation**, **class weighting**, **early stopping**
- Performed **fine-tuning** using a custom dataset of 100 images (captured via iPhone 14 Pro Max)
- Fine-tuned only the top 15–30 layers with a reduced learning rate (`1e-4` → `1e-5`)

---

## 🔧 Project Highlights

- ✅ Transfer learning using MobileNetV2 (ImageNet-pretrained)
- ✅ Mid-network customization using Conv2D blocks
- ✅ Two-stage training:
  1. On FER2013 (general facial emotion learning)
  2. On manual dataset (personal fine-tuning)
- ✅ Powerful data augmentation
- ✅ Real-time webcam prediction using OpenCV
- ✅ Final model saved as `final_emotion_model.h5`

---

## 📊 Dataset Summary

| Dataset        | Images | Description |
|----------------|--------|-------------|
| FER2013        | ~30K   | Public, grayscale, resized to 224×224 |
| Manual Dataset | 100    | Personal iPhone photos (20/class), RGB |

---

## 📈 Model Performance

| Stage                  | Accuracy | Notes                      |
|------------------------|----------|----------------------------|
| FER2013 Training       | 85.6%    | After 20 epochs            |
| FER2013 Validation     | 72.6%    |                            |
| FER2013 Test           | 72.8%    |                            |
| Fine-Tuning Accuracy   | ↑        | Improved face-specific accuracy in real-time detection |

---

## ▶️ Real-Time Detection

Run emotion detection on webcam feed using the trained model.

### Requirements

```bash
pip install tensorflow opencv-python numpy


---

## ▶️ Run

```bash
python realtime_emotion_detection.py


The webcam will open, detect faces, and display live predictions for one of the following emotions:
neutral, happy, sad, angry, surprise



📂 Files in the Repository
| File                              | Description                        |
| --------------------------------- | ---------------------------------- |
| `train_emotion_model.ipynb`       | Training on FER2013                |
| `train_final_emotion_model.ipynb` | Fine-tuning on personal dataset    |
| `emotion_model.h5`                | Base model after FER2013 training  |
| `final_emotion_model.h5`          | Final model after fine-tuning      |
| `realtime_emotion_detection.py`   | Live detection using webcam        |
| `README.md`                       | Project documentation              |
| `fer2013_dataset.zip`             | Original dataset                   |
| `manual_dataset.zip` *(optional)* | Personal training data (if public) |

---
📌 Notes
Fine-tuning with manual data improved accuracy for the author's face, but reduced generalization for other faces.

Model is optimized for a personal use-case (emotion tracking, webcam interfaces, etc.)

Further improvement can be achieved with more diverse personal images.

---
👤 Author
Youssef Osama Fawzy
GitHub • LinkedIn
