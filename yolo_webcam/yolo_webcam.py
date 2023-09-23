from ultralytics import YOLO
import cv2
import cvzone
import math


# cam = 0  # for webcam
# cam = 'http://xxx.xxx.xxx.xxx:xxxx/video'  # for droid cam
cam = '../Videos/bikes.mp4'  # for videos
cap = cv2.VideoCapture(cam)  # for videos
# cap.set(3, 1280)
# cap.set(4, 720)
# cap.set(10, 100)

model = YOLO('../yolo_weights/yolov8n.pt')
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
while True:
    success, img = cap.read()
    # img = cv2.flip(img, 1)  # mirror the input stream
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h), colorR=(0, 0, 255), colorC=(0, 255, 0))

            # Confidence
            conf = math.ceil((box.conf[0]*100))/100
            # print(conf)
            # class Name
            cls = box.cls[0]
            cvzone.putTextRect(img, f'{className[int(cls)]} {conf}', (max(0, x1), max(40, y1)), scale=1, thickness=1)


    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
