import os
import random
import torch

import cv2
from ultralytics import YOLO

from tracker import Tracker


video_path = os.path.join('.', 'data', 'road2.mp4')
video_out_path = os.path.join('.', 'out.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

model = YOLO("yolov8s.pt")
accident_model = YOLO("runs/detect/train6/weights/best.pt")

# Print model architecture
print(model)

class_names = {0: 'person',
 1: 'bicycle',
 2: 'car',
 3: 'motorcycle',
 4: 'airplane',
 5: 'bus',
 6: 'train',
 7: 'truck',
 8: 'boat',
 9: 'traffic light',
 10: 'fire hydrant',
 11: 'stop sign',
 12: 'parking meter',
 13: 'bench',
 14: 'bird',
 15: 'cat',
 16: 'dog',
 17: 'horse',
 18: 'sheep',
 19: 'cow',
 20: 'elephant',
 21: 'bear',
 22: 'zebra',
 23: 'giraffe',
 24: 'backpack',
 25: 'umbrella',
 26: 'handbag',
 27: 'tie',
 28: 'suitcase',
 29: 'frisbee',
 30: 'skis',
 31: 'snowboard',
 32: 'sports ball',
 33: 'kite',
 34: 'baseball bat',
 35: 'baseball glove',
 36: 'skateboard',
 37: 'surfboard',
 38: 'tennis racket',
 39: 'bottle',
 40: 'wine glass',
 41: 'cup',
 42: 'fork',
 43: 'knife',
 44: 'spoon',
 45: 'bowl',
 46: 'banana',
 47: 'apple',
 48: 'sandwich',
 49: 'orange',
 50: 'broccoli',
 51: 'carrot',
 52: 'hot dog',
 53: 'pizza',
 54: 'donut',
 55: 'cake',
 56: 'chair',
 57: 'couch',
 58: 'potted plant',
 59: 'bed',
 60: 'dining table',
 61: 'toilet',
 62: 'tv',
 63: 'laptop',
 64: 'mouse',
 65: 'remote',
 66: 'keyboard',
 67: 'cell phone',
 68: 'microwave',
 69: 'oven',
 70: 'toaster',
 71: 'sink',
 72: 'refrigerator',
 73: 'book',
 74: 'clock',
 75: 'vase',
 76: 'scissors',
 77: 'teddy bear',
 78: 'hair drier',
 79: 'toothbrush',
 80: 'accident'
 }

trackers = [Tracker() for i in range(len(class_names))]

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(100)]

detection_threshold = 0.5
while ret:

    results = model(frame)
    accident_results = accident_model(frame)
    
    for accident_result in accident_results:
        detections = [[] for i in range(len(class_names))]
        for r in accident_result.boxes.data.tolist():
            x1_acc, y1_acc, x2_acc, y2_acc, score, class_id = r
            x1_acc = int(x1)
            x2_acc = int(x2)
            y1_acc = int(y1)
            y2_acc = int(y2)
            class_id_accident = int(class_id) + 80
            if score > detection_threshold:
                detections[class_id_accident].append([x1_acc, y1_acc, x2_acc, y2_acc, score])
            
        for i in range(len(class_names)):
            trackers[i].update(frame, detections[i])
        
        for tracker_id, tracker in enumerate(trackers):
            for track in tracker.tracks:
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                track_id = track.track_id
                class_id_accident = tracker_id

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[class_id_accident]), 3)
                class_name = class_names[class_id]
                cv2.putText(frame, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    for result in results:
        detections = [[] for i in range(len(class_names))]
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold:
                detections[class_id].append([x1, y1, x2, y2, score])
        
        for i in range(len(class_names)):
            trackers[i].update(frame, detections[i])

        for tracker_id, tracker in enumerate(trackers):
            for track in tracker.tracks:
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                track_id = track.track_id
                class_id = tracker_id

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[class_id]), 3)
                class_name = class_names[class_id]
                cv2.putText(frame, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cap_out.write(frame)
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()
