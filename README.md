

---

# Driver Drowsiness Detection System — VigilEye Web App

A deep learning–based computer vision project that detects driver fatigue by analyzing eye states (Awake vs Sleepy). The system is deployed as a lightweight web application for real-time inference.

---

# Project Overview

Driver drowsiness is a major contributor to road accidents. This project builds a real-time ready deep learning pipeline that classifies eye images into **Awake** or **Sleepy** and serves predictions through a FastAPI web application.

The project covers the full pipeline:

* Model training and evaluation
* Performance optimization
* Backend deployment (FastAPI)
* Browser-based frontend interface

---

# Objectives

Automated Detection
Build a binary classifier to distinguish between Awake and Sleepy eye states.

Performance Optimization
Compare a custom CNN with a transfer learning model.

Efficiency
Reduce training time using a high-speed data loading method.

Scalability
Deploy a lightweight model suitable for edge devices and web inference.

---

# Technology Stack

Language: Python
Deep Learning: TensorFlow / Keras
Data Processing: NumPy, Pandas, PIL
Visualization: Matplotlib, Scikit-learn
Backend: FastAPI
Frontend: HTML, CSS, JavaScript
Environment: Google Colab (GPU)

---

# Key Features

* Real-time ready image size: 96 × 96
* Data augmentation for robustness
* Ultra-fast training via local VM SSD transfer (~100× faster)
* Automatic saving of models and evaluation charts to Google Drive
* Lightweight architecture suitable for deployment
* Browser-based inference via FastAPI

---

# Models Implemented

## 1. Custom CNN (Baseline)

Architecture:

* 3 Conv2D layers
* BatchNormalization
* MaxPooling
* Dropout (0.5)
* Dense Sigmoid classifier

Result:
Accuracy ≈ 74%

---

## 2. MobileNetV2 Transfer Learning Model

Architecture:

* MobileNetV2 base (frozen)
* GlobalAveragePooling
* Dropout (0.2)
* Sigmoid output layer

Result:

* Accuracy: 88.56%
* Higher recall and better generalization

---

# Evaluation Metrics (MobileNetV2)

| Metric            | Value  |
| ----------------- | ------ |
| Accuracy          | 88.56% |
| Precision         | 0.8678 |
| Recall            | 0.9059 |
| F1-Score          | 0.8864 |
| Balanced Accuracy | 0.8859 |

Predictions Summary:

* Correct Predictions: 1757
* Incorrect Predictions: 227

---

# How the Model Works

1. Data Acquisition
   Images from the MRL Eye Dataset are labeled as Awake or Sleepy.

2. Preprocessing
   Images resized to 96 × 96 and normalized.

3. Feature Extraction
   MobileNetV2 extracts eye patterns such as:

* Edges
* Eye shape
* Eyelid position

4. Classification

| Probability | Output |
| ----------- | ------ |
| < 0.5       | Awake  |
| > 0.5       | Sleepy |

5. Inference
   Model predicts driver state instantly from a single image.

---

# VigilEye — Drowsiness Detection Web App

## Quick Start

### 1. Backend (FastAPI)

Install dependencies:

```bash
pip install fastapi uvicorn tensorflow pillow numpy python-multipart
```

Place `mobilenet_eye_model.h5` in the same folder as `main.py`, then run:

```bash
uvicorn main:app --reload --port 8000
```

Backend runs at:
[http://localhost:8000](http://localhost:8000)

API documentation:
[http://localhost:8000/docs](http://localhost:8000/docs)

---

### 2. Frontend

Open in browser:

```
http://localhost:8000
```

No build step required. The frontend is pure HTML/CSS/JavaScript served by FastAPI.

---

# Project Structure

```
├── index.html
├── backend/
│   ├── main.py
│   └── mobilenet_eye_model.h5
```

---

# Demo Login

Username: admin
Password: vigil123

---

# Application Flow

1. User Login
2. Accept Terms (10k dataset subset disclaimer)
3. Upload eye image
4. Receive prediction (AWAKE / SLEEPY)

---

# Model Details (Deployed Version)

Architecture: MobileNetV2 (transfer learning)
Training Dataset: 10,000 images (subset of 84k MRL Eye Dataset)
Validation Accuracy: 88.56%
Recall: 90.59%
Input Size: 96 × 96 RGB

---

# Future Improvements

* Real-time webcam streaming detection
* Alarm/alert system for fatigue detection
* Mobile and edge deployment (Raspberry Pi / Android)
* Full driver monitoring system integration

---

# Conclusion

The transfer learning approach significantly improved accuracy while keeping the model lightweight and fast. The VigilEye web application demonstrates a complete end-to-end pipeline for real-time driver drowsiness detection.
