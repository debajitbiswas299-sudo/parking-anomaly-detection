# python detect_parking.py query_video.mp4 model.pkl output.mp4


import cv2
import numpy as np
import argparse
import pickle
from ultralytics import YOLO


def extract_features(box):
    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    bottom_x = (x1 + x2) / 2.0
    bottom_y = y2

    aspect_ratio = height / width if width > 0 else 0

    return [bottom_x, bottom_y, width, height, aspect_ratio]


def main(input_video, model_path, output_video):

    model = YOLO("yolov8n.pt")

    with open(model_path, "rb") as f:
        kde, scaler = pickle.load(f)

    cap = cv2.VideoCapture(input_video)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    threshold = -7.5   # Tune this

    print("Running anomaly detection...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.45, classes=[1, 3], verbose=False)

        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)

                feat = np.array([extract_features(box)])
                feat_scaled = scaler.transform(feat)

                score = kde.score_samples(feat_scaled)[0]

                is_anomaly = score < threshold

                color = (0, 0, 255) if is_anomaly else (0, 255, 0)
                label = f"WRONG ({score:.2f})" if is_anomaly else f"OK ({score:.2f})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)

    cap.release()
    out.release()

    print("✅ Detection complete:", output_video)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("model")
    parser.add_argument("output")
    args = parser.parse_args()

    main(args.input, args.model, args.output)

