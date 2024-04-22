
import streamlit as st
import cv2
import mtcnn
import numpy as np
import face_recognition
from datetime import datetime

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
cap = cv2.VideoCapture(0)  # Initialize cap here

# Load known encodings and names
irtiza_encodings = np.load('irtiza_encodings.npy')
mateen_encodings = np.load('mateen_encodings.npy')
known_encodings = np.concatenate((irtiza_encodings, mateen_encodings))

# Load names from respective files
with open('irtiza_names.txt', 'r') as f:
    irtiza_names = [line.strip() for line in f]

with open('mateen_names.txt', 'r') as f:
    mateen_names = [line.strip() for line in f]

known_names = irtiza_names + mateen_names

# Initialize facespresent dictionary to keep track of recognized faces
facespresent = {}

# Main Streamlit app
st.title("Face Recognition App")

# Create a placeholder for the frame to be displayed
frame_placeholder = st.empty()

# Streamlit loop
while True:
    ret, frame = cap.read()
    try:
        check_image(frame)
    except ValueError as e:
        print(e)
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
                # Check if facespresent is already marked for this person
                if name not in facespresent:
                    facespresent[name] = True
                    # Mark facespresent
                    mark_facespresent(name)
                    st.write(f"{name} marked facespresent at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        else:
            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Update the placeholder with the new frame
    frame_placeholder.image(frame, channels="BGR")

# Save facespresent dictionary to a file
    np.save('facespresent.npy', facespresent)



