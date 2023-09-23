import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *


cam = '../Videos/cars.mp4'  # for videos
cap = cv2.VideoCapture(cam)  # for videos

model = YOLO('../yolo_weights/yolov8l.pt')
className = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread('mask.png')

# tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [400, 297, 673, 297]
total_car_count = []

while True:
    success, img = cap.read()
    img_region = cv2.bitwise_and(img, mask)
    imgGraphic = cv2.imread('graphics.png', cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphic, (0, 0))

    results = model(img_region, stream=True)
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1

            # Confidence
            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            current_class = className[cls]
            if (current_class == "car" or current_class == "truck" or current_class == "bus"
                    or current_class == "motorbike" and conf > 0.3):
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    tracker_results = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 4)

    for result in tracker_results:
        x1, y1, x2, y2, ID = result
        x1, y1, x2, y2, ID = int(x1), int(y1), int(x2), int(y2), int(ID)
        # print(result)
        w, h = x2 - x1, y2-y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=10, t=2, colorR=(255, 0, 255), colorC=(0, 0, 255), rt=1)
        cvzone.putTextRect(img, f'{ID}', (max(0, x1), max(40, y1)), scale=1,
                           thickness=2, colorT=(255, 255, 255), colorR=(0, 0, 255), offset=5)
        cx, cy = x1+w//2, y1 + h//2  # center points
        # cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1]-15 < cy < limits[1]+15:
            if total_car_count.count(ID) == 0:
                total_car_count.append(ID)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 4)
                cvzone.cornerRect(img, (x1, y1, w, h), l=10, t=2, colorR=(0, 255, 255), colorC=(0, 255, 0), rt=1)
                cvzone.putTextRect(img, f'{ID}', (max(0, x1), max(40, y1)), scale=1,
                                   thickness=2, colorT=(255, 255, 255), colorR=(0, 255, 0), offset=5)

    cv2.putText(img, str(len(total_car_count)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 8)
    cv2.imshow("Counting Cars", img)

    # cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

