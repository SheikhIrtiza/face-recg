import cv2
import mtcnn

class Webcam:
    '''
    This class encapsulates the functionality related to accessing the webcam and capturing frames from it.

    Attributes:
    --------------
    camera_index:
        An integer representing the index of the webcam to be used. Default is 0, which typically corresponds to the primary webcam.
    cap: 
        A cv2.VideoCapture object representing the webcam capture device.

    Methods:
    --------------
    __init__(self, camera_index=0):
        The constructor initializes the Webcam object with the specified camera_index and creates a cv2.VideoCapture object for webcam access.
    __del__(self):
        The destructor releases the webcam capture device when the object is destroyed.
    capture_frame(self):
        Method to capture a frame from the webcam using cap.read() and return a tuple (ret, frame) indicating whether a frame was successfully captured (ret) and the captured frame (frame).
    '''
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index)

    def __del__(self):
        self.cap.release()

    def capture_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

class FaceDetector:
    '''
    This class encapsulates the functionality related to detecting faces in an image using the MTCNN (Multi-Task Cascaded Convolutional Neural Network) model.

    Attributes:
    --------------
    detector:
        An instance of the MTCNN detector.

    Methods:
    --------------
    __init__(self):
        The constructor initializes the FaceDetector object and creates an instance of the MTCNN detector.
    detect_faces(self, frame):
        Method to detect faces in the given frame using the MTCNN detector. It takes a single argument frame (image) and returns a list of detected faces.
    '''
    def __init__(self):
        self.detector = mtcnn.MTCNN()

    def detect_faces(self, frame):
        return self.detector.detect_faces(frame)

class FaceDetectionApp:
    '''
    This class represents the face detection application, coordinating the webcam capture and face detection processes.

    Attributes:
    --------------
    webcam:
        An instance of the Webcam class for webcam access.
    face_detector:
        An instance of the FaceDetector class for face detection.

    Methods:
    --------------
    __init__(self, camera_index=0):
        The constructor initializes the FaceDetectionApp object with a specified camera_index (default is 0). It creates instances of Webcam and FaceDetector.
    run(self):
        Method to start the face detection application loop. It continuously captures frames from the webcam, detects faces in each frame, draws bounding boxes around the detected faces, and displays the frames in a window using OpenCV. It terminates when the 'q' key is pressed.
    draw_face_boxes(self, frame, result_list):
        Helper method to draw bounding boxes around detected faces on the given frame.
    display_frame(self, window_name, frame):
        Helper method to display the given frame in a window with the specified window_name.
    '''
    def __init__(self, camera_index=0):
        self.webcam = Webcam(camera_index)
        self.face_detector = FaceDetector()

    def run(self):
        while True:
            ret, frame = self.webcam.capture_frame()
            if not ret:
                print("Failed to capture frame")
                break

            faces = self.face_detector.detect_faces(frame)
            self.draw_face_boxes(frame, faces)
            print(f"Detected {len(faces)} faces")

            self.display_frame('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def draw_face_boxes(self, frame, result_list):
        for result in result_list:
            x, y, width, height = result['box']
            confidence = result['confidence']
            if confidence > 0.9:
                cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)

    def display_frame(self, window_name, frame):
        cv2.imshow(window_name, frame)

def run_face_detection():
    app = FaceDetectionApp()
    app.run()

