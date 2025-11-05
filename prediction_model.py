#!/usr/bin/env python3
import sys
import os
import pandas as pd
import joblib
from jetson_inference import poseNet
from jetson_utils import videoSource, videoOutput

# ------------------------------------------------------------
# Load trained SVM, scaler, and label encoder
# ------------------------------------------------------------
MODEL_PATH = "/jetson-inference/data/project/pose_input/svm_pose_model.pkl"
SCALER_PATH = "/jetson-inference/data/project/pose_input/svm_pose_scaler.pkl"
ENCODER_PATH = "/jetson-inference/data/project/pose_input/svm_pose_label_encoder.pkl"

print("trained SVM model...")
svm_model   = joblib.load(MODEL_PATH)
scaler      = joblib.load(SCALER_PATH)
label_enc   = joblib.load(ENCODER_PATH)
print("load and scaler loaded.\n")

# ============================================================
# Define fixed PoseNet BODY keypoints (must match training)
# ============================================================
KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "neck"
]

# ============================================================
# Get image path from command line
# ============================================================
if len(sys.argv) < 2:
    print("Usage: python3 predict_pose_svm_static.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.isfile(image_path):
    print(f"image not found: {image_path}")
    sys.exit(1)

# ============================================================
# Load PoseNet model
# ============================================================
net = poseNet("resnet18-body", sys.argv, threshold=0.15)
input_stream = videoSource(image_path, argv=sys.argv)
img = input_stream.Capture()

poses = net.Process(img)
if len(poses) == 0:
    print("Person not detected")
    sys.exit(0)

pose = poses[0]  # take first detected person

# ============================================================
# Extract keypoint features
# ============================================================
features = {kp: ["", "", ""] for kp in KEYPOINTS}

for k in pose.Keypoints:
    name = net.GetKeypointName(k.ID)
    if name in features:
        conf = getattr(k, "confidence", getattr(k, "C", 0.0))
        features[name] = [round(k.x, 2), round(k.y, 2), round(conf, 3)]

# Flatten features into one row
flat = []
for kp in KEYPOINTS:
    vals = features[kp]
    vals = [0 if v == "" else v for v in vals]  # blanks -> 0
    flat.extend(vals)

sample_df = pd.DataFrame(
    [flat],
    columns=[f"{kp}_{a}" for kp in KEYPOINTS for a in ["x", "y", "conf"]]
)
print(sample_df)
# ============================================================
# Scale features and run SVM prediction
# ============================================================
sample_scaled = scaler.transform(sample_df)
pred = svm_model.predict(sample_scaled)[0]
prob = svm_model.predict_proba(sample_scaled)[0].max()
label = label_enc.inverse_transform([pred])[0]

# ============================================================
# Output result
# ============================================================
print("Pose C lassification Result")
print("------------------------------")
print(f"Image     : {os.path.basename(image_path)}")
print(f"Prediction: {label}")
print(f"Confidence: {prob:.2f}")
