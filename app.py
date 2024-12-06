import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tempfile
import os

# Load model configuration files
config_file = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = "frozen_inference_graph.pb"

# Load the model
model = cv2.dnn_DetectionModel(frozen_model, config_file)

# Load class labels
filename = "labels.txt"
classLabels = []
with open(filename, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

# Set the model input size and configurations
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# Function for image object detection
def detect_objects_in_image(img):
    font_scale = 3
    font = cv2.FONT_HERSHEY_PLAIN
    ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.55)
    
    for ClassIndex, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
        x, y, w, h = boxes
        label = classLabels[ClassIndex - 1]
        st.write(f"Detected class: {label}, Confidence: {conf}")  # Debugging line
        
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
        cv2.putText(img, label, (x + 10, y + 40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Function for video object detection
def detect_objects_in_video(video_file):
    # Create a temporary file to store the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        temp_video_path = temp_video.name

    # Open the video file
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        st.error("Cannot open video")
        return

    # Process each frame and display it with detected objects
    font_scale = 1
    font = cv2.FONT_HERSHEY_PLAIN
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # If frame is not read correctly, break the loop
        
        # Detect objects in the frame
        ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)
        if len(ClassIndex) > 0:
            for ClassIndex, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
                # Draw rectangle
                cv2.rectangle(frame, (boxes[0], boxes[1]), (boxes[0] + boxes[2], boxes[1] + boxes[3]), (255, 0, 0), 2)
                # Add text (class label)
                label = classLabels[ClassIndex - 1]
                cv2.putText(frame, label, (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale, color=(0, 255, 0), thickness=2)

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame with Streamlit
        st.image(frame_rgb, channels="RGB", use_column_width=True)
        
    cap.release()
    os.remove(temp_video_path)  # Delete the temporary video file after processing

# Streamlit UI elements for Introduction, Image Detection, and Video Detection
def main():
    # Sidebar for slide selection
    slide = st.sidebar.radio("Choose a slide", ("Introduction", "Image Detection", "Video Detection"))

    if slide == "Introduction":
        show_introduction()

    elif slide == "Image Detection":
        show_image_detection()

    elif slide == "Video Detection":
        show_video_detection()

# Show Introduction Section
def show_introduction():
    st.title("Traffic Monitoring Using Object Detection")
    st.write("""
        **Urban Traffic Monitoring:**
        In urban areas, traffic monitoring plays a crucial role in maintaining safety, improving traffic flow, and managing congestion. 
        With advancements in computer vision and machine learning, we can now automate traffic monitoring using cameras and object detection algorithms. 
        This project aims to use object detection to monitor and analyze traffic in real-time.

        **Objective:**
        The system detects and tracks various objects (vehicles, pedestrians, etc.) in live traffic video feeds. 
        It provides useful data like vehicle count, traffic density, and movement patterns.
        
        **How the Object Detection System Works:**
        - **Input:** Video feed from traffic cameras (live or pre-recorded).
        - **Output:** Detected objects with bounding boxes and real-time statistics
        - **Workflow:** 
          1. Image/Video Preprocessing
          2. Object Detection (using pre-trained models like YOLO or SSD)
          3. Post-processing (tracking objects and generating data)
          4. Displaying Results
        
        **Challenges Faced:**
        - Variability in lighting and environmental conditions.
        - Object occlusion (vehicles/pedestrians partially blocked).
        - Real-time processing requirements.
        
        **Solutions:**
        - Data augmentation for training with diverse conditions.
        - Optimized object detection models like YOLOv3 and SSD for real-time performance.""")

    st.title("OpenCV in Object Detection:")
    st.write("""
        OpenCV (Open Source Computer Vision Library) is an open-source library containing over 2,500 optimized algorithms for real-time computer vision applications.
        It is widely used for tasks like image processing, video analysis, and object detection.
        
        **How OpenCV Helps in Object Detection:**
        - OpenCV allows for real-time video processing, which is crucial for traffic monitoring applications.
        - It supports pre-trained models such as YOLO, SSD, and Faster R-CNN, which are ideal for detecting vehicles, pedestrians, and other objects in traffic monitoring systems.
        
        **Object Detection in Traffic Monitoring:**
        Object detection is the key technology behind automated traffic monitoring systems. It enables the identification of various objects (vehicles, pedestrians, etc.) 
        in a live video feed, providing critical insights into traffic conditions. In this system, a deep learning model (like YOLO or SSD) analyzes each frame of the video 
        and detects objects by drawing bounding boxes.
        
    """)

# Show Image Detection Section
def show_image_detection():
    st.title("Image Object Detection")
    st.write("Upload an image for object detection.")
    
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Detecting objects...")
        image = np.array(image)
        result_image = detect_objects_in_image(image)
        st.image(result_image, caption="Detected Image", use_column_width=True)

# Show Video Detection Section
def show_video_detection():
    st.title("Video Object Detection")
    st.write("Upload a video for object detection.")
    
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "webm", "mov"])
    if uploaded_video is not None:
        detect_objects_in_video(uploaded_video)

if __name__ == "__main__":
    main()
