import cv2
import numpy as np

# Load YOLO model (YOLOv3)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO class names
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Set input image size and scale factor
input_size = (416, 416)
scale_factor = 1.0 / 255.0

# Read an image
image = cv2.imread("sample.jpg")

# Preprocess the image
blob = cv2.dnn.blobFromImage(image, scale_factor, input_size, swapRB=True, crop=False)
net.setInput(blob)

# Get YOLO output layer names
output_layer_names = net.getUnconnectedOutLayersNames()

# Forward pass and get detections
detections = net.forward(output_layer_names)

# Process detections
for detection in detections:
    for obj in detection:
        scores = obj[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x, center_y, width, height = map(int, obj[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]]))
            x, y = center_x - width // 2, center_y - height // 2
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(image, classes[class_id], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the result
cv2.imshow("YOLO Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
