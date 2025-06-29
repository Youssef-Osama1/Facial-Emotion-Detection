import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("emotion_model.h5")


class_names = np.load("emotion_labels.npy", allow_pickle=True)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


IMG_SIZE = 224


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face = frame[y:y+h, x:x+w]

        try:

            face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face_normalized = face_resized / 255.0
            face_input = np.expand_dims(face_normalized, axis=0)

            preds = model.predict(face_input, verbose=0)
            emotion = class_names[np.argmax(preds)]

            cv2.putText(frame, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        except:
            pass 

    cv2.imshow('Real-Time Emotion Detection', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()