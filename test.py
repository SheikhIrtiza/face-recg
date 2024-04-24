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

def extract_and_save_encoding(frame, name):
    faces = detect_faces(detector, frame)
    if len(faces) == 0:
        st.warning("No faces detected. Please make sure your face is visible.")
        return

    # Display the captured frame with face boxes
    draw_facebox(frame, faces)
    st.image(frame, channels="BGR", caption="Detected Faces")

    if len(faces) == 1:
        selected_face = faces[0]
    else:
        st.write("Multiple faces detected. Please select the face to extract encoding:")
        selected_face_index = st.selectbox("Select Face", range(len(faces)))
        selected_face = faces[selected_face_index]

    x, y, width, height = selected_face['box']
    face_img = extract_face(frame, (x, y, width, height))
    face_encoding = face_recognition.face_encodings(face_img)

    if face_encoding:
        np.save(f'encodings/{name}_encodings.npy', face_encoding)
        st.success(f"Face encoding for {name} extracted and saved successfully!")
    else:
        st.error("Failed to extract face encoding.")

# Initialize MTCNN detector
detector = mtcnn.MTCNN()

# Main Streamlit app
st.title("Face Recognition App")

# Create a placeholder for the frame to be displayed
frame_placeholder = st.empty()

# Create a checkbox to trigger face extraction
extract_face_checkbox = st.button("Extract Face Encoding")

# Streamlit loop
cap = cv2.VideoCapture(0)  # Initialize webcam
while extract_face_checkbox:
    ret, frame = cap.read()
    try:
        check_image(frame)
    except ValueError as e:
        st.error(e)
        continue

    if st.button("Extract and Save Encoding"):
        name = st.text_input("Enter your name:")
        if name:
            extract_and_save_encoding(frame, name)

    # Update the placeholder with the new frame
    frame_placeholder.image(frame, channels="BGR")

# Release the webcam
cap.release()
cv2.destroyAllWindows()
