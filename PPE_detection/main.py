from ultralytics import YOLO
import cv2
import cvzone
import math


# cam = 0  # for webcam
# cam = 'http://xxx.xxx.xxx.xxx:xxxx/video'  # for droid cam
cam = '../Videos/ppe-1.mp4'  # for videos
cap = cv2.VideoCapture(cam)  # for videos
# cap.set(3, 1280)
# cap.set(4, 720)
# cap.set(10, 100)

model = YOLO('ppe.pt')
className = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person',
             'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery', 'mini-van', 'sedan',
             'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']
myColor = (0, 0, 255)
while True:
    success, img = cap.read()
    if success is False:
        break
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
            # cvzone.cornerRect(img, (x1, y1, w, h), colorR=(0, 0, 255), colorC=(0, 255, 0))
            # Confidence
            conf = math.ceil((box.conf[0]*100))/100
            # print(conf)
            # class Name
            cls = int(box.cls[0])
            current_class = className[cls]
            if conf > 0.5:
                if current_class == 'Hardhat' or current_class == 'Gloves' or current_class == 'Mask' or current_class == 'Safety Vest':
                    myColor = (0, 255, 0)
                elif current_class == 'NO-Hardhat' or current_class == 'NO-Mask' or current_class == 'NO-Safety Vest':
                    myColor = (0, 0, 255)
                else:
                    myColor = (255, 0, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, thickness=3)
                cvzone.putTextRect(img, f'{className[int(cls)]} {conf}', (max(0, x1), max(40, y1)),
                                   scale=1, thickness=1, colorB=myColor, colorT=(255, 255, 255), colorR=myColor, offset=5)


    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
