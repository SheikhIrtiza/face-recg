import cv2
import mtcnn

# print(cv2.getBuildInformation())

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

while True:
    ret, frame = cap.read()
    try:
        check_image(frame)
    except ValueError as e:
        print(e)
        continue

    faces = detect_faces(detector, frame)
    draw_facebox(frame, faces)
    print(f"Detected {len(faces)} faces")

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): #(continue)
        break

    # Check for 'q' key press only if faces are detected (stop)
    # if len(faces) > 0:
    #     if cv2.waitKey(0) & 0xFF == ord('q'):
    #         break


cap.release()
cv2.destroyAllWindows()



