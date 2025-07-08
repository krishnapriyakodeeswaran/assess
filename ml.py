import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open video file
video = cv2.VideoCapture("vehicledet.mp4")  # change this to your video name

# Get width, height, and fps of the video
width = int(video.get(3))
height = int(video.get(4))
fps = int(video.get(5))

# Create video writer to save output
output = cv2.VideoWriter("outputdetect.mp4", cv2.VideoWriter_fourcc(*"MP4V"), fps, (width, height))

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Detect objects in the frame
    results = model(frame)
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("Detection", annotated_frame)     

    # Save to output file
    output.write(annotated_frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
output.release()
cv2.destroyAllWindows()
print("✅ Video detection done and saved!")
