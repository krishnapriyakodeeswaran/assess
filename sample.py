import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

# Load the YOLOv8n model
model = YOLO("yolov8n.pt")

def detect_image(image_path, output_path="output_image.jpg"):
    # Read the image
    image = cv2.imread("/workspaces/codespaces-blank/detect.jfif")
    if image is None:
        print(" Error: Image not found.")
        return
    # Run detection
    results = model(image)
    result = results[0]

    # Draw bounding boxes, labels, and confidence scores
    annotated = result.plot()
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]
        print(f"Detected: {class_name} ({confidence:.2%})")

    # displays the output image
    cv2.imwrite(output_path, annotated)
    print(f"✅ Detection complete! Saved as: {output_path}")

   
    # Optional: Print detected class names and confidence scores


# Call the function with your image path
detect_image("detect.jfif")
