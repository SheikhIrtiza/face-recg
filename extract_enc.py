##extracts encodings one by one from faces

import cv2
import mtcnn
import numpy as np
import face_recognition
# from datetime import datetime.


def extract_face(frame, face_location):
    x, y, width, height = face_location
    face_img = frame[y:y+height, x:x+width]
    return face_img

def draw_facebox(frame, result_list):
    for result in result_list:
        x, y, width, height = result['box']
        confidence = result['confidence']
        if confidence > 0.9:
            cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)

def detect_faces(detector, frame):
    faces = detector.detect_faces(frame)
    return faces

def check_image(frame):
    if frame is None:
        raise ValueError("Invalid image captured from the webcam.")

# Initialize MTCNN detector
detector = mtcnn.MTCNN()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Mode 0: Face Encoding
print("Face Encoding Mode")
name = input("Enter your name: ")
known_encodings = []
known_names = []

while True:
    ret, frame = cap.read()
    try:
        check_image(frame)
    except ValueError as e:
        print(e)
        continue

    faces = detect_faces(detector, frame)
    draw_facebox(frame, faces)

    # Extract face encodings and names
    for face in faces:
        x, y, width, height = face['box']
        face_img = extract_face(frame, (x, y, width, height))
        face_encoding = face_recognition.face_encodings(face_img)
        if face_encoding:
            known_encodings.extend(face_encoding)
            known_names.extend([name]*len(face_encoding))

    cv2.imshow('Face Encoding', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save known face encodings and names
np.save(f'encodings\{name}_encodings.npy', known_encodings)
with open(f'encodings\{name}_names.txt', 'w') as f:

    for name in known_names:
        f.write(f"{name}\n")

cap.release()
cv2.destroyAllWindows()
