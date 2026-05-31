import cv2
from ultralytics import YOLO

# Load pre-trained YOLO model
model = YOLO("yolov8n.pt")

# Video file path
cap = cv2.VideoCapture("test_video.mp4")

# Check video opened or not
if not cap.isOpened():
    print("Error: test_video.mp4 file not found or cannot be opened.")
    exit()

# Vehicle class IDs: car, motorcycle, bus, truck
vehicle_classes = [2, 3, 5, 7]

cv2.namedWindow("Vehicle Detection and Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Vehicle Detection and Tracking", 600, 1000)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video ended.")
        break

    # Detect and track vehicles
    results = model.track(
        frame,
        persist=True,
        classes=vehicle_classes,
        conf=0.25,
        tracker="bytetrack.yaml"
    )

    # Draw bounding boxes, labels and tracking IDs
    annotated_frame = results[0].plot()

    # Show resized output window
    cv2.imshow("Vehicle Detection and Tracking", annotated_frame)

    # Slow down video speed
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()