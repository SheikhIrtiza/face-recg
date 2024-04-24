import streamlit as st
import cv2
import mtcnn
import numpy as np
import face_recognition
from datetime import datetime

global counter
counter = 0

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

def mark_facespresent(name):
    with open('facespresent.csv', 'a') as f:
        now = datetime.now()
        dt_string = now.strftime('%d-%m-%y,%H:%M:%S')
        f.write(f"{name},{dt_string}\n")

# Initialize MTCNN detector
detector = mtcnn.MTCNN()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Streamlit app
st.title("Face Recognition and Encoding App")

# Option for extracting and saving face encodings

st.header("Extract and Save Face Encodings")
name_for_encoding = st.text_input("Enter your name for encoding:")
counter = 0
start_encoding = st.button("Start Encoding", key = f"start_encoding_button{counter}")
frame_placeholder = st.empty()  # Placeholder for the frame

encoding_file = None
names_file = None

if start_encoding and name_for_encoding:
    encoding_file = f'encodings/{name_for_encoding}_encodings.npy'
    names_file = f'encodings/{name_for_encoding}_names.txt'

    while True:
        ret, frame = cap.read()
        try:
            check_image(frame)
        except ValueError as e:
            st.error(e)
            continue

        faces = detect_faces(detector, frame)
        draw_facebox(frame, faces)

        # Extract and save face encodings
        for face in faces:
            x, y, width, height = face['box']
            face_img = extract_face(frame, (x, y, width, height))
            face_encoding = face_recognition.face_encodings(face_img)
            if face_encoding:
                np.save(encoding_file, face_encoding)
                with open(names_file, 'a') as f:
                    f.write(f"{name_for_encoding}\n")
                st.success(f"Face encoding for {name_for_encoding} extracted and saved successfully!")
                frame_placeholder.image(frame, channels="BGR", caption="Detected Faces")
                break

        if st.button("Stop Encoding", key = f"stop_encoding_button_1{counter}"):
            break
        
# Option for recognizing faces
st.header("Face Recognition")
start_recognition = st.button("Start Recognition", key = f"start_recognition_button{counter}")

if start_recognition:
    known_encodings = []
    known_names = []

    # Load known encodings and names
    irtiza_encodings = np.load('encodings/irtiza_encodings.npy')
    mateen_encodings = np.load('encodings/mateen_encodings.npy')
    known_encodings.extend(irtiza_encodings)
    known_encodings.extend(mateen_encodings)

    with open('encodings/irtiza_names.txt', 'r') as f:
        irtiza_names = [line.strip() for line in f]

    with open('encodings/mateen_names.txt', 'r') as f:
        mateen_names = [line.strip() for line in f]

    known_names.extend(irtiza_names)
    known_names.extend(mateen_names)

    facespresent = {}

    while True:
        ret, frame = cap.read()
        try:
            check_image(frame)
        except ValueError as e:
            st.error(e)
            continue

        faces = detect_faces(detector, frame)
        draw_facebox(frame, faces)

        # Recognize faces
        for face in faces:
            x, y, width, height = face['box']
            face_img = extract_face(frame, (x, y, width, height))
            face_encoding = face_recognition.face_encodings(face_img)
            if face_encoding:
                matches = face_recognition.compare_faces(known_encodings, face_encoding[0])
                name = "Unknown"
                if True in matches:
                    match_index = matches.index(True)
                    name = known_names[match_index]
                    # Mark faces present
                    if name not in facespresent:
                        facespresent[name] = True
                        mark_facespresent(name)
                        st.write(f"{name} marked present at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        frame_placeholder.image(frame, channels="BGR", caption="Recognized Faces")

        if st.button("Stop Recognition", key = f"stop_recognition_button_1{counter}"):
            break

cap.release()
cv2.destroyAllWindows()
