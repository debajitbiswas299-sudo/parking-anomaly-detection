# python train_parking.py normal_video.mp4 model.pkl

import cv2
import numpy as np
import argparse
import pickle
import os
from ultralytics import YOLO
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler


def extract_features(box):
    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    bottom_x = (x1 + x2) / 2.0
    bottom_y = y2

    aspect_ratio = height / width if width > 0 else 0

    return [bottom_x, bottom_y, width, height, aspect_ratio]


def main(input_folder, output_model):

    model = YOLO("yolov8n.pt")

    all_features = []

    video_files = [f for f in os.listdir(input_folder)
                   if f.endswith((".mp4", ".avi", ".mov"))]

    print(f"Found {len(video_files)} training videos")

    for video_name in video_files:

        path = os.path.join(input_folder, video_name)
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            print("Skipping:", video_name)
            continue

        print("Processing:", video_name)

        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1

            # 🔥 sample every 4 frames
            if frame_id % 4 != 0:
                continue

            results = model(frame, conf=0.45, classes=[1, 3], verbose=False)

            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()

                for box in boxes:
                    feat = extract_features(box)
                    all_features.append(feat)

        cap.release()

    all_features = np.array(all_features)

    print("Total collected samples:", len(all_features))

    # Normalize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)

    # KDE model
    kde = KernelDensity(kernel='gaussian', bandwidth=0.8)
    kde.fit(features_scaled)

    with open(output_model, "wb") as f:
        pickle.dump((kde, scaler), f)

    print("✅ Model trained and saved:", output_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("output_model")
    args = parser.parse_args()

    main(args.input_folder, args.output_model)


