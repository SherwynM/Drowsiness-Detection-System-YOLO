from ultralytics import YOLO
from pygame import mixer
from datetime import datetime
from csv import writer as wr
import cv2
import os

# Load Haar Cascades for face, left eye, and right eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
leye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
reye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

# Load YOLO model
yolo_model = YOLO('models/best.pt')

mixer.init()
sound = mixer.Sound('alarm.wav')

path = os.getcwd()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# Initialize detection results and state
state = "Unknown"
frame_count = 0
detection_frequency = 8  # Run YOLO every 8 frames for efficiency
score = 0
csv_file = 'analysis.csv'

#Initialize the file
with open(csv_file, mode='w', newline='') as file:
    writer = wr(file)
    writer.writerow(["ID", "Timestamp", "State"])

id = 0  # ID in the csv file

# Helper function to perform YOLO prediction and label assignment
def yolo_predict_and_label(eye_crop, position):
    yolo_results = yolo_model.predict(eye_crop)
    local_state = "Unknown"
    for detection in yolo_results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = detection
        if conf > 0.5:
            label = yolo_model.names[int(cls)]
            local_state = "Open" if label == "Open Eyes" else "Closed"
            cv2.putText(frame, f'{label} {conf:.2f}', position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return local_state

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (fx, fy, fw, fh) in faces:
        face_roi_gray = gray[fy:fy + fh // 2, fx:fx + fw]
        face_roi_color = frame[fy:fy + fh // 2, fx:fx + fw]

        # Detect eyes
        left_eye = leye_cascade.detectMultiScale(face_roi_gray)
        right_eye = reye_cascade.detectMultiScale(face_roi_gray)

        # Run YOLO only on specific frames
        if frame_count % detection_frequency == 0:
            for (lx, ly, lw, lh) in left_eye:
                left_eye_crop = cv2.resize(face_roi_color[ly:ly + lh, lx:lx + lw], (20, 20))
                if left_eye_crop.size > 0:
                    state = yolo_predict_and_label(left_eye_crop, (fx + lx, fy + ly - 10))

            for (rx, ry, rw, rh) in right_eye:
                right_eye_crop = cv2.resize(face_roi_color[ry:ry + rh, rx:rx + rw], (20, 20))
                if right_eye_crop.size > 0:
                    state = yolo_predict_and_label(right_eye_crop, (fx + rx, fy + ry - 10))

    # Display eye state
    cv2.putText(frame, f'State: {state}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if state == "Closed":
        score += 1
    else:
        score = max(0, score - 1)

    # Display score
    cv2.putText(frame, f'Score: {score}', (10, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 255), 1, cv2.LINE_AA)

    # Trigger alarm if score exceeds threshold
    if score >= 30:
        score = 30
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        
        try:
            sound.play()
        except Exception as e:
            print(f"Error playing sound: {e}")

    id += 1
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S") + f".{now.microsecond // 1000:03d}"
    with open(csv_file, mode='a', newline='') as file:
        writer = wr(file)
        writer.writerow([id, timestamp, state])

    cv2.imshow('Live Video Feed', frame)

    # Increase frame count
    frame_count = (frame_count + 1) % 1000
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
