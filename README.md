# 🚗 Parking Anomaly Detection System

## 📌 Overview
This project presents an AI-based system for detecting **wrong parking** using computer vision and anomaly detection techniques.

The system uses **YOLOv8** for object detection and **Kernel Density Estimation (KDE)** to identify abnormal parking patterns.

---

## 🧠 Technologies Used
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Scikit-learn (KDE)

---

## ⚙️ How It Works
1. Detect vehicles using YOLOv8  
2. Extract features (position, width, height, aspect ratio)  
3. Train KDE model on normal parking  
4. Detect anomalies based on probability score  
5. Label output as:
   - 🟢 OK (correct parking)
   - 🔴 WRONG (incorrect parking)

---

## ▶️ How to Run

### 🔹 Training
```bash
python train_parking.py dataset_folder model.pkl
