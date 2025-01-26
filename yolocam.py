import cv2
import numpy as np
import time
import os

# Load YOLO
try:
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
classes = []
try:
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
except Exception as e:
    print(f"Error loading class names: {e}")
    exit()

font = cv2.FONT_HERSHEY_PLAIN

# Define confidence and NMS thresholds
conf_threshold = 0.5
nms_threshold = 0.4
# Create a named window and resize it
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 800, 600)

# Define a function to adjust confidence threshold
def set_conf_threshold(x):
    global conf_threshold
    conf_threshold = x / 100

# Define a function to adjust NMS threshold
def set_nms_threshold(x):
    global nms_threshold
    nms_threshold = x / 100

# Create trackbars for dynamic threshold adjustment
cv2.createTrackbar("Confidence Threshold", "Image", int(conf_threshold * 100), 100, set_conf_threshold)
cv2.createTrackbar("NMS Threshold", "Image", int(nms_threshold * 100), 100, set_nms_threshold)

# Ensure the output directory exists
output_dir = "detected_objects"
os.makedirs(output_dir, exist_ok=True)

def apply_attention(frame, boxes, scores, attention_threshold=0.3):
    attention_map = np.zeros(frame.shape[:2], dtype=np.float32)
    for box, score in zip(boxes, scores):
        x, y, w, h = box
        attention_map[y:y+h, x:x+w] += score

    attention_map = cv2.GaussianBlur(attention_map, (15, 15), 0)
    attention_map = cv2.normalize(attention_map, None, 0, 1, cv2.NORM_MINMAX)
    
    attention_frame = frame.copy()
    mask = attention_map < attention_threshold
    attention_frame[mask] = (0, 0, 0)
    return attention_frame

def detect_objects(frame):
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    detected_objects = {}
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 1, color, 2)
            
            if label in detected_objects:
                detected_objects[label] += 1
            else:
                detected_objects[label] = 1

            if x >= 0 and y >= 0 and x + w <= width and y + h <= height:
                detected_object_image = frame[y:y+h, x:x+w]
                if detected_object_image.size > 0:
                    filename = f"{output_dir}/{label}_{int(time.time())}.jpg"
                    cv2.imwrite(filename, detected_object_image)

            if label == "person":
                cv2.putText(frame, "ALERT: Person Detected", (10, height - 50), font, 2, (0, 0, 255), 3)

    attention_frame = apply_attention(frame, boxes, confidences)

    return attention_frame, detected_objects

def process_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return
    
    starting_time = time.time()
    frame_id = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            frame_id += 1

            attention_frame, detected_objects = detect_objects(frame)

            elapsed_time = time.time() - starting_time
            fps = frame_id / elapsed_time
            cv2.putText(attention_frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 2, (0, 255, 0), 3)

            y_offset = 100
            for obj, count in detected_objects.items():
                cv2.putText(attention_frame, f"{obj}: {count}", (10, y_offset), font, 2, (0, 255, 0), 3)
                y_offset += 50

            cv2.imshow("Image", attention_frame)
            
            key = cv2.waitKey(1)
            if key == 27:  # ESC key to break
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Could not read image.")
        return
    attention_frame, detected_objects = detect_objects(frame)

    y_offset = 50
    for obj, count in detected_objects.items():
        cv2.putText(attention_frame, f"{obj}: {count}", (10, y_offset), font, 2, (0, 255, 0), 3)
        y_offset += 50

    cv2.imshow("Image", attention_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    starting_time = time.time()
    frame_id = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_id += 1
            attention_frame, detected_objects = detect_objects(frame)

            elapsed_time = time.time() - starting_time
            fps = frame_id / elapsed_time
            cv2.putText(attention_frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 2, (0, 255, 0), 3)

            y_offset = 100
            for obj, count in detected_objects.items():
                cv2.putText(attention_frame, f"{obj}: {count}", (10, y_offset), font, 2, (0, 255, 0), 3)
                y_offset += 50

            cv2.imshow("Image", attention_frame)
            
            key = cv2.waitKey(1)
            if key == 27:  # ESC key to break
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    input_type = input("Enter input type (camera/image/video): ").strip().lower()
    if input_type == "camera":
        process_camera()
    elif input_type == "image":
        image_path = input("Enter image path: ").strip()
        process_image(image_path)
    elif input_type == "video":
        video_path = input("Enter video path: ").strip()
        process_video(video_path)
    else:
        print("Invalid input type. Please enter 'camera', 'image', or 'video'.")

if __name__ == "__main__":
    main()
