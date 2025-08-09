import cv2
import face_recognition
import numpy as np
import pickle

with open("face_recognition_model.pkl", "rb") as f:
    clf = pickle.load(f)

video_capture = cv2.VideoCapture(0)

print("[INFO] Starting live face recognition... Press 'q' to exit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("[ERROR] Failed to capture frame.")
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    
        name = "Unknown"
        try:
            name = clf.predict([face_encoding])[0]
        except Exception as e:
            print("[ERROR] Prediction failed:", e)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Live Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
