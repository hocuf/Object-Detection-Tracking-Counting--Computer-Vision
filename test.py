"""

ARCHITECTURE

- Import libraries (start)
- Get the capture (real time or video)
- Return and run the capture
- Finish it and keep going
- Process the capture in loop
    - Object detection and tracking
        - Load YOLO and run it
        - get results
            - access to data
            - process to data
        - show results

"""


# - Import libraries (start)
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


from ultralytics import YOLO
model = YOLO('yolov8l.pt')

# - Get the capture (real time or video)
video_path = 'data/test_video.mp4'
cap = cv2.VideoCapture(video_path)

# - Return and run the capture
while cap.isOpened():
    count = 0
    ret, frame = cap.read()
    if ret:
        # results = model.track(frame, persist=True)
        # plotting = results[0].plot()

        results =model(frame, verbose=False)
        labels = results[0].names

        for i, box in enumerate(results[0].boxes.xyxy):
            x1, y1, x2, y2 = map(int, box[:4])
            score = float(results[0].boxes.conf[i])
            cls = int(results[0].boxes.cls[i])
            name = labels[cls]
            #print(score)
            if (name== 'car') and (score > 0.4):
                count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                text = f'ID: {cls}, SCORE:{score: .2f}'
                cv2.putText(frame, text, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.putText(frame, f'Total Cars: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Window', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()