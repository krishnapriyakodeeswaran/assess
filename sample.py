import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

model = YOLO("yolov8n.pt")
def detect_image(image_path, output_path="output_image.jpg"):
    image = cv2.imread("/workspaces/codespaces-blank/detect.jfif")
    if image is None:
        print(" Error: Image not found.")
        return
    results = model(image)
    result = results[0]
    annotated = result.plot()
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]
        print(f"Detected: {class_name} ({confidence:.2%})")
    cv2.imwrite(output_path, annotated)
    print(f" Detection complete! Saved as: {output_path}")
detect_image("detect.jfif")
