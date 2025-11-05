#!/usr/bin/env python3
import sys
import argparse
import csv
import os
from glob import glob
from jetson_inference import poseNet
from jetson_utils import videoSource

# -----------------------------------------------------------
# Fixed header list for PoseNet BODY model (18 keypoints)
# -----------------------------------------------------------
KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "neck"
]

# -----------------------------------------------------------
# Command-line Arguments
# -----------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Extract PoseNet keypoints for all images and save aligned CSV with fixed headers",
    formatter_class=argparse.RawTextHelpFormatter,
    epilog=poseNet.Usage()
)

parser.add_argument("input_folder", type=str, help="Folder containing input images")
parser.add_argument("--network", type=str, default="resnet18-body", help="Pre-trained model to load")
parser.add_argument("--threshold", type=float, default=0.15, help="Minimum detection threshold")
parser.add_argument("--csv", type=str, default="/jetson-inference/data/project/pose_input/pose_results.csv", help="CSV filename to append to")
parser.add_argument("--pose_label", type=str, default=None, help="Pose label (default: folder name)")

args = parser.parse_args()

# -----------------------------------------------------------
# Auto-set pose label from folder name if not provided
# -----------------------------------------------------------
if args.pose_label is None:
    args.pose_label = os.path.basename(os.path.normpath(args.input_folder)).replace("_", " ")
print(f"Pose label set to: {args.pose_label}")

# -----------------------------------------------------------
# Load model
# -----------------------------------------------------------
net = poseNet(args.network, sys.argv, args.threshold)

# -----------------------------------------------------------
# Collect input images
# -----------------------------------------------------------
image_paths = sorted(
    glob(os.path.join(args.input_folder, "*.jpg")) +
    glob(os.path.join(args.input_folder, "*.jpeg")) +
    glob(os.path.join(args.input_folder, "*.png"))
)
if not image_paths:
    print(f"No images found in {args.input_folder}")
    sys.exit(1)

print(f"Found {len(image_paths)} images")

# -----------------------------------------------------------
# Prepare CSV
# -----------------------------------------------------------
file_exists = os.path.isfile(args.csv)
csv_file = open(args.csv, mode="a", newline="")
writer = csv.writer(csv_file)

# Build header once
if not file_exists:
    header = ["image_name"]
    for kp in KEYPOINTS:
        header += [f"{kp}_x", f"{kp}_y", f"{kp}_conf"]
    header += ["pose_label"]
    writer.writerow(header)

# -----------------------------------------------------------
# Process each image
# -----------------------------------------------------------
for img_path in image_paths:
    print(f"Processing: {img_path}")
    input_stream = videoSource(img_path, argv=sys.argv)
    img = input_stream.Capture()
    if img is None:
        print(f"Skipped (could not read): {img_path}")
        continue

    poses = net.Process(img)
    if len(poses) == 0:
        print(f"No poses detected in {img_path}")
        continue

    for pose in poses:
        # Initialize all keypoints as blank
        kp_dict = {kp: ["", "", ""] for kp in KEYPOINTS}

        # Fill detected keypoints
        for k in pose.Keypoints:
            name = net.GetKeypointName(k.ID)
            if name in kp_dict:
                conf = getattr(k, "confidence", getattr(k, "C", 0.0))
                kp_dict[name] = [round(k.x, 2), round(k.y, 2), round(conf, 3)]

        # Build row strictly in header order
        row = [os.path.basename(img_path)]
        for kp in KEYPOINTS:
            row += kp_dict[kp]  # fill detected or blank values
        row.append(args.pose_label)

        writer.writerow(row)

csv_file.close()
print(f"All results saved and perfectly aligned → '{args.csv}'")
