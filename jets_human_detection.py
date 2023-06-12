# Import required libraries
import cv2
import numpy as np
import argparse
import time
import pyudev
import subprocess

# Install darknet:
# git clone https://github.com/pjreddie/darknet.git
# cd darknet
# make


# Create argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--webcam', help="True/False", default=False)
parser.add_argument('--play_video', help="True/False", default=False)
parser.add_argument('--image', help="True/False", default=False)
parser.add_argument('--video_path', help="Path of video file", default="videos/car_on_road.mp4")
parser.add_argument('--image_path', help="Path of image to detect objects", default="Images/bicycle.jpg")
parser.add_argument('--verbose', help="To print statements", default=True)
args = parser.parse_args()

# Function to print available cameras
def print_available_cameras():
    command = 'v4l2-ctl --list-devices'
    output = subprocess.check_output(command, shell=True).decode()

    camera_names = []
    device_lines = output.strip().split('\n')
    for i, line in enumerate(device_lines):
        if line.startswith('\t/dev/video'):
            camera_name = device_lines[i-1].strip()
            camera_names.append(camera_name)

    num_cameras = len(camera_names)
    print(f"Number of available cameras: {num_cameras}")

    for i, camera_name in enumerate(camera_names):
        print(f"{i}: {camera_name}")


# Load YOLO model
def load_yolo():
    # Load YOLO model weights and configuration
    net = cv2.dnn.readNet("darknet/yolov3.weights", "darknet/cfg/yolov3.cfg")
    
    # Load classes for object detection
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines() if line.strip() == "person"]
    
    # Get output layers and generate random colors for each class
    output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    return net, classes, colors, output_layers

# Function to start webcam
def start_webcam(resolution=(320, 240), fps=10, camera_index=0):
    # Create VideoCapture object for specified camera index
    cap = cv2.VideoCapture(camera_index)

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    # Set desired FPS
    cap.set(cv2.CAP_PROP_FPS, fps)

    return cap

# Function to display blob
def display_blob(blob):
    '''
    Display images for each channel (RED, GREEN, BLUE)
    '''
    for b in blob:
        for n, imgb in enumerate(b):
            cv2.imshow(str(n), imgb)

# Function to detect objects in an image
def detect_objects(img, net, outputLayers):
    # Generate blob from input image
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    
    # Set the input for the neural network
    net.setInput(blob)
    
    # Forward pass through the network
    outputs = net.forward(outputLayers)
    
    return blob, outputs

# Function to get bounding box dimensions of detected objects
def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []

    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.1 and class_id == 0:  # Filter for "person" class
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)

    return boxes, confs, class_ids

# Function to draw labels on the image
def draw_labels(boxes, confs, colors, class_ids, classes, img):
    font = cv2.FONT_HERSHEY_PLAIN
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    else:
        cv2.putText(img, "No people detected", (10, 30), font, 1, (0, 0, 255), 2)
        
    cv2.imshow("Image", img)
    
    
def image_detect(img_path):
    model, classes, colors, output_layers = load_yolo()  # Load YOLO model
    image, height, width, channels = load_image(img_path)  # Load image
    blob, outputs = detect_objects(image, model, output_layers)  # Detect objects in the image
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)  # Get bounding box dimensions
    draw_labels(boxes, confs, colors, class_ids, classes, image)  # Draw labels on the image
    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break

def start_video(video_path):
    model, classes, colors, output_layers = load_yolo()  # Load YOLO model
    cap = cv2.VideoCapture(video_path)  # Open video file
    while True:
        _, frame = cap.read()  # Read frame from the video
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)  # Detect objects in the frame
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)  # Get bounding box dimensions
        draw_labels(boxes, confs, colors, class_ids, classes, frame)  # Draw labels on the frame
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()  # Release video file

def webcam_detect(camera_index):
    model, classes, colors, output_layers = load_yolo()  # Load YOLO model
    cap = start_webcam(camera_index=camera_index)  # Start webcam
    timer_start = time.time()
    image_counter = 0
    frame_counter = 0
    unique_ids = set()

    while True:
        _, frame = cap.read()  # Read frame from the webcam
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)  # Detect objects in the frame
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)  # Get bounding box dimensions
        draw_labels(boxes, confs, colors, class_ids, classes, frame)  # Draw labels on the frame
        key = cv2.waitKey(1)

        if time.time() - timer_start > 300:  # Capture and process image every 5 minutes
            image_path = f"image_{image_counter}.jpg"
            cv2.imwrite(image_path, frame)  # Save image
            print(f"Saved image: {image_path}")
            image_counter += 1
            timer_start = time.time()

            saved_image = cv2.imread(image_path)  # Load saved image
            saved_height, saved_width, saved_channels = saved_image.shape
            blob, outputs = detect_objects(saved_image, model, output_layers)  # Detect objects in saved image
            saved_boxes, saved_confs, saved_class_ids = get_box_dimensions(outputs, saved_height, saved_width)  # Get bounding box dimensions

            unique_ids.clear()
            for i in range(len(saved_boxes)):  # Count number of unique people in the saved image
                if saved_class_ids[i] == 0:  # Check if class_id is 0 for people
                    unique_ids.add(saved_class_ids[i])

            count = len(unique_ids)
            print(f"Number of people detected in saved image: {count}")

        frame_counter += 1
        if frame_counter % 10 == 0:  # Calculate FPS every 10 frames
            elapsed_time = time.time() - timer_start
            fps = frame_counter / elapsed_time
            print(f"FPS: {fps:.2f}")

        if key == 27:
            break

    cap.release()  # Release webcam

if __name__ == '__main__':
    webcam = args.webcam
    video_play = args.play_video
    image = args.image

    if webcam:
        if args.verbose:
            print('---- Starting Web Cam object detection ----')
        print_available_cameras()  # Print available cameras
        camera_index = int(input("Enter the index of the camera to use: "))
        webcam_detect(camera_index)  # Start webcam object detection

