from ultralytics import YOLO
import cv2


model = YOLO('../yolo_weights/yolov8m.pt')
results = model('Images/3.png', show=True)
cv2.waitKey(0)


