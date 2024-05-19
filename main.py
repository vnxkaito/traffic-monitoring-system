import os
import random
import cv2
from ultralytics import YOLO

from tracker import Tracker

counter = 0
global_tracker = []
video_path = os.path.join('.', 'data', 'road2.mp4')
video_out_path = os.path.join('.', 'out.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

left_margin = frame.shape[1] * 0.01
right_margin = frame.shape[1] * 0.99
top_margin = frame.shape[0] * 0.99
bottom_margin = frame.shape[0] * 0.01

model = YOLO("yolov8n.pt")

tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(1)]

detection_threshold = 0.4
while ret:

    results = model(frame)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold and class_id in [2, 3, 5, 7]:
                detections.append([x1, y1, x2, y2, score])

        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id
            if(track_id not in global_tracker and x1>left_margin and x2<right_margin and y1>bottom_margin and y2<top_margin):
                counter+=1
                global_tracker.append([track_id, 0])

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
            cv2.putText(frame, f'Number of vehicles: {counter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        

        
        for idx, t in enumerate(global_tracker):
            if t not in [track.track_id for track in tracker.tracks]:
                global_tracker[idx][0] = -1

    cv2.imshow('frame', frame)
    cv2.waitKey(5)

    cap_out.write(frame)
    ret, frame = cap.read()

print(counter)
cap.release()
cap_out.release()
cv2.destroyAllWindows()
