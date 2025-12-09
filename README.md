# 🧍‍♀️ Jetson  Pose Detection – Phase 1 🤖  
### *T-Pose Classification using Jetson Inference + SVM*

<p align="center">
  <!-- Badges Row -->
  <img src="https://img.shields.io/badge/Python-3.8-blue?logo=python&logoColor=white" alt="Python 3.8"/>
  <img src="https://img.shields.io/badge/Jetson-Nano-green?logo=nvidia&logoColor=white" alt="Jetson Nano"/>
  <img src="https://img.shields.io/badge/Accuracy-97%25-success?logo=dependabot&logoColor=white" alt="Model Accuracy"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?logo=open-source-initiative&logoColor=white" alt="License MIT"/>
</p>

<p align="center">
  <img src="pose_data/test_images/Jetson.png" alt="Jetson  Demo" width="720"/>
</p>

**Jetson ** is an edge AI project that leverages **NVIDIA Jetson Inference PoseNet** to detect human body poses and classify them using a Support Vector Machine (SVM).  
This first phase focuses on identifying **T-Pose** vs **Not-T-Pose** using keypoints generated from PoseNet and stored for model training.

---

## 🧩 Project Overview

✅ **PoseNet** – extracts 18 body keypoints from each frame.  
✅ **CSV Logging** – saves keypoint coordinates & confidence values.  
✅ **SVM Training** – learns to classify T-Pose vs Not-T-Pose.  
✅ **Real-Time Prediction** – runs inference with saved `.pkl` models.  

<p align="center">
  <img src="pose_data/test_images/Architecture.png" alt="Jetson  System Architecture" width="800"/>
</p>

---

## ⚙️ Folder Structure

```bash
jetson-pose-detection/
├── data/
│   └── pose_keypoints.csv
├── pose_data/
│   ├── Tpose/
│   ├── Not_Tpose/
│   └── test_images/
│       ├── Architecture.png
│       ├── Jetson.png
│       ├── Screenshot from 2025-11-04 23-27-56.png
│       └── Screenshot from 2025-11-04 23-29-28.png
├── svm_model/
│   ├── svm_pose_model.pkl
│   ├── svm_pose_label_encoder.pkl
│   └── svm_pose_scaler.pkl
├── Data_collection.py        # Extracts PoseNet keypoints, saves CSV
├── Model_training.py         # Trains SVM model
├── prediction_model.py       # Runs real-time inference
├── LICENSE
└── README.md
```

---

## 🧠 PoseNet Keypoints Used

```python
KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "neck"
]
```

Each frame produces `(x, y, confidence)` values for these 18 keypoints.

---

## 🚀 How to Run

### 1️⃣ Collect Keypoints Data
Run your data collection script to extract keypoints and save them to CSV:

```bash
python3 Data_collection.py
```

Generates:
```
data/pose_keypoints.csv
```

---

### 2️⃣ Train the SVM Model
Train the classifier using scikit-learn:
```bash
python3 Model_training.py
```

Saves model files to:
```
svm_model/
 ├── svm_pose_model.pkl
 ├── svm_pose_label_encoder.pkl
 └── svm_pose_scaler.pkl
```

---

### 3️⃣ Real-Time Pose Classification
Run pose prediction on an image or live input:
```bash
python3 prediction_model.py /path/to/image.png
```

- PoseNet extracts keypoints  
- Scaler normalizes them  
- SVM predicts **T-Pose** or **Not-T-Pose**  
- Outputs prediction + confidence  

---

## 🧾 Example Jetson Terminal Outputs

### ✅ **T-Pose Detected**
<p align="center">
  <img src="pose_data/test_images/Screenshot from 2025-11-04 23-27-56.png" alt="T-Pose Classification Result" width="800"/>
</p>

### ❌ **Not-T-Pose Detected**
<p align="center">
  <img src="pose_data/test_images/Screenshot from 2025-11-04 23-29-28.png" alt="Not T-Pose Classification Result" width="800"/>
</p>

---

## 📊 Example CSV Data

| nose_x | nose_y | left_shoulder_x | right_shoulder_x | left_hip_x | right_hip_x | neck_x | neck_y | label |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0.421 | 0.331 | 0.312 | 0.632 | 0.310 | 0.621 | 0.470 | 0.320 | Tpose |
| 0.422 | 0.475 | 0.310 | 0.600 | 0.309 | 0.590 | 0.469 | 0.460 | Not_Tpose |

---

## 📈 Model Evaluation

| Metric | Score |
|:--|:--:|
| Accuracy | 0.97 ✅ |
| Precision | 0.95 |
| Recall | 0.96 |
| F1-Score | 0.95 |

---

## 🧭 Roadmap

| Phase | Description | Status |
|:--|:--|:--:|
| 1 | Keypoint Detection + SVM (T-Pose / Not-T-Pose) | ✅ Completed |
| 2 | Multi-Pose Classification (Squat, Plank, Jump) | 🚧 In Progress |
| 3 | Real-Time Voice Feedback | 🔜 Planned |
| 4 | Streamlit Dashboard for Analytics | 🌐 Future |
| 5 | Edge Optimization (TensorRT) | 💡 Upcoming |

---

## 💡 Use Cases

- 🧘‍♀️ **Fitness & Posture Tracking**  
- 🧑‍🏫 **AI-Assisted Exercise Coaching**  
- 🎮 **Gesture-Based Controls**  
- 🧠 **Human Motion Analytics**

---

## 🧡 Credits

Developed by **Anjali Jha**  
M.S. in Data Science — **University of Maryland, Baltimore County (UMBC)**  

- Edge AI: [NVIDIA Jetson Inference](https://github.com/dusty-nv/jetson-inference)  
- Classifier: [scikit-learn SVM](https://scikit-learn.org/stable/modules/svm.html)  
- Dataset: Captured with Jetson Orin PoseNet inference  

---

## 📜 License

MIT License © 2025 [Anjali Jha](https://github.com/Anjali9815)

---

<p align="center">
  <b>“Every pose is a datapoint toward better movement insight.”</b>
</p>
