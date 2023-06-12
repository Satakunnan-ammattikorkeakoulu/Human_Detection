import cv2
import numpy as np
import argparse
import time
import objc
from AppKit import AVCaptureDevice

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--webcam', help="True/False", default=False)
parser.add_argument('--play_video', help="True/False", default=False)
parser.add_argument('--image', help="True/False", default=False)
parser.add_argument('--video_path', help="Path of video file", default="videos/car_on_road.mp4")
parser.add_argument('--image_path', help="Path of image to detect objects", default="Images/bicycle.jpg")
parser.add_argument('--verbose', help="To print statements", default=True)
args = parser.parse_args()

def print_available_cameras():
    # Get available cameras
    devices = AVCaptureDevice.devices()
    camera_names = []

    # Iterate over devices and find cameras
    for device in devices:
        if device.hasMediaType_('vide'):
            camera_name = f"{device.localizedName()} ({device.uniqueID()})"
            camera_names.append(camera_name)

    num_cameras = len(camera_names)
    print(f"Number of available cameras: {num_cameras}")

    # Print the list of cameras
    for i, camera_name in enumerate(camera_names):
        print(f"{i}: {camera_name}")

# Load YOLO model
def load_yolo():
    # Load YOLO weights and configuration
    net = cv2.dnn.readNet("darknet/yolov3.weights", "darknet/cfg/yolov3.cfg")

    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines() if line.strip() == "person"]

    # Get output layer names
    output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]

    # Generate random colors for each class
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    return net, classes, colors, output_layers


def start_webcam(resolution=(320, 240), fps=10, camera_index=0):
    # Open webcam
    cap = cv2.VideoCapture(camera_index)

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    # Set desired FPS
    cap.set(cv2.CAP_PROP_FPS, fps)

    return cap


def display_blob(blob):
    
    #Display images for each channel (RED, GREEN, BLUE)
    for b in blob:
        for n, imgb in enumerate(b):
            cv2.imshow(str(n), imgb)


def detect_objects(img, net, outputLayers):
    # Create blob from the image
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)

    # Set the blob as input to the network
    net.setInput(blob)

    # Forward pass through the network
    outputs = net.forward(outputLayers)

    return blob, outputs


def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []

    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]

            # Filter for "person" class and confidence threshold
            if conf > 0.1 and class_id == 0:
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


def draw_labels(boxes, confs, colors, class_ids, classes, img):
    font = cv2.FONT_HERSHEY_PLAIN

    # Apply non-maximum suppression to remove overlapping bounding boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]

            # Draw bounding box and label on the image
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    else:
        # If no people detected, display a message
        cv2.putText(img, "No people detected", (10, 30), font, 1, (0, 0, 255), 2)

    # Display the image with bounding boxes and labels
    cv2.imshow("Image", img)


def image_detect(img_path):
    model, classes, colors, output_layers = load_yolo()

    # Load image and get its dimensions
    image, height, width, channels = load_image(img_path)

    # Detect objects in the image
    blob, outputs = detect_objects(image, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)

    # Draw bounding boxes and labels on the image
    draw_labels(boxes, confs, colors, class_ids, classes, image)

    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break


def start_video(video_path):
    model, classes, colors, output_layers = load_yolo()

    # Open video file
    cap = cv2.VideoCapture(video_path)

    while True:
        # Read a frame from the video
        _, frame = cap.read()
        height, width, channels = frame.shape

        # Detect objects in the frame
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)

        # Draw bounding boxes and labels on the frame
        draw_labels(boxes, confs, colors, class_ids, classes, frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()


def webcam_detect(camera_index):
    model, classes, colors, output_layers = load_yolo()

    # Start the webcam
    cap = start_webcam(camera_index=camera_index)

    timer_start = time.time()
    image_counter = 0
    frame_counter = 0
    unique_ids = set()

    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape

        # Detect objects in the frame
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)

        # Draw bounding boxes and labels on the frame
        draw_labels(boxes, confs, colors, class_ids, classes, frame)

        key = cv2.waitKey(1)

        if time.time() - timer_start > 5:  # 300 seconds = 5 minutes
            # Save an image every 5 minutes
            image_path = f"image_{image_counter}.jpg"
            cv2.imwrite(image_path, frame)
            print(f"Saved image: {image_path}")
            image_counter += 1
            timer_start = time.time()

            # Load the saved image
            saved_image = cv2.imread(image_path)
            saved_height, saved_width, saved_channels = saved_image.shape
            blob, outputs = detect_objects(saved_image, model, output_layers)
            saved_boxes, saved_confs, saved_class_ids = get_box_dimensions(outputs, saved_height, saved_width)

            # Count the number of unique people in the saved image
            unique_ids.clear()
            for i in range(len(saved_boxes)):
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

    cap.release()


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--webcam', help="True/False", default=False)
    parser.add_argument('--play_video', help="True/False", default=False)
    parser.add_argument('--image', help="True/False", default=False)
    parser.add_argument('--video_path', help="Path of video file", default="videos/car_on_road.mp4")
    parser.add_argument('--image_path', help="Path of image to detect objects", default="Images/bicycle.jpg")
    parser.add_argument('--verbose', help="To print statements", default=True)
    args = parser.parse_args()

    webcam = args.webcam
    video_play = args.play_video
    image = args.image

    if webcam:
        if args.verbose:
            print('---- Starting Web Cam object detection ----')
        print_available_cameras()
        camera_index = int(input("Enter the index of the camera to use: "))
        webcam_detect(camera_index)
