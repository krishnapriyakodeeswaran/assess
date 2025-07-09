YOLOv8 Object Detection with OpenCV (Image + Video)

This project demonstrates how to perform object detection using [YOLOv8] with OpenCV in Python. It supports:

i)Image detection
ii)Video Detection

Requirements:
Install the required Python libraries:
pip install ultralytics opencv-python matplotlib numpy
 Image Detection
 Code File: sample.py
This script loads an image and detects objects using a pre-trained YOLOv8n model.

Usage
 Placing the  image file (detect.jfif) in the same directory.

Run the script:
python sample.py
 Output
Annotated image saved as: output_image.jpg

->Detected classes and confidence printed in terminal

 Video Detection
 Code File: ml.py
This script runs real-time detection on a video file and saves the output video with bounding boxes.

Usage
Placing the video file (vehicledet.mp4) in the same folder.

Run the script:
python ml.py
 Output
Annotated video saved as: output.mp4

Note for GitHub Codespaces
GUI functions like cv2.imshow() are not supported.

Use mp4v codec instead of MP4V:
cv2.VideoWriter_fourcc(*"mp4v")
 Model
We use the pre-trained YOLOv8n model provided by Ultralytics. It is fast and ideal for real-time inference.

File Structure:
1)detect_image.py       
2)detect_video.py      
3)detect.jfif          
4)output_image.jpg    
5)outputdetect.mp4            
6)yolov8n.pt           
7)README.md

Output Example
Detected: person (99.23%)
Detected: traffic light (87.56%)
Detection complete! Saved as: output_image.jpg

 Author
Krishnapriya
College of Engineering, Guindy
M.Sc. Integrated IT
July 2025

