#!/usr/bin/env python3
# svm_pose_classifier.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# ------------------------------------------------------------
# 1. Load the dataset
# ------------------------------------------------------------
df = pd.read_csv("/jetson-inference/data/project/pose_input/pose_results.csv")

# Drop non-numeric columns
df = df.drop(columns=["image_name"], errors="ignore")

# Fill blanks or NaNs with 0 (missing keypoints)
df = df.fillna(0)

# Separate features and labels
X = df.drop(columns=["pose_label"])
y = df["pose_label"]

# Encode labels (T Pose -> 1, Not T Pose -> 0)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ------------------------------------------------------------
# 2. Split data into train/test
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
)

# ------------------------------------------------------------
# 3. Normalize features
# ------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# 4. Train SVM classifier
# ------------------------------------------------------------
svm_model = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# ------------------------------------------------------------
# 5. Evaluate model
# ------------------------------------------------------------
y_pred = svm_model.predict(X_test_scaled)
print("Model Evaluation Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# ------------------------------------------------------------
# 6. Save model and scaler
# ------------------------------------------------------------
joblib.dump(svm_model, "/jetson-inference/data/project/pose_input/svm_pose_model.pkl")
joblib.dump(scaler, "/jetson-inference/data/project/pose_input/svm_pose_scaler.pkl")
joblib.dump(le, "/jetson-inference/data/project/pose_input/svm_pose_label_encoder.pkl")

print("\nModel, scaler, and label encoder saved successfully!")
