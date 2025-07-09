import cv2
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
video = cv2.VideoCapture("vehicledet.mp4")  # change this to your video name

width = int(video.get(3))
height = int(video.get(4))
fps = int(video.get(5))
output = cv2.VideoWriter("outputdetect.mp4", cv2.VideoWriter_fourcc(*"MP4V"), fps, (width, height))
while True:
    ret, frame = video.read()
    if not ret:
        break
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("Detection", annotated_frame)     
    output.write(annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
output.release()
cv2.destroyAllWindows()
print(" Video detection done and saved!")
