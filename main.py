##detection on images 


import matplotlib.pyplot as plt
from mtcnn import MTCNN

def draw_facebox(filename, result_list):
    # Load the image
    data = plt.imread(filename)
    # Plot the image
    plt.imshow(data)
    # Get the context for drawing boxes
    ax = plt.gca()
    # Initialize counter for faces
    num_faces = 0
    # Plot each box
    for result in result_list:
        # Get coordinates and confidence score
        x, y, width, height = result['box']
        confidence = result['confidence']
        # Filter out low-confidence detections
        if confidence > 0.9:  # Adjust confidence threshold as needed
            # Draw the box
            rect = plt.Rectangle((x, y), width, height, fill=False, color='green')
            ax.add_patch(rect)
            # Increment face counter
            num_faces += 1
    # Show the plot
    plt.show()
    return num_faces

def detect_faces(filename):
    # Load the image
    pixels = plt.imread(filename)
    # Initialize MTCNN detector
    detector = MTCNN()
    # Detect faces in the image
    faces = detector.detect_faces(pixels)
    return faces

# Filename of the image
filename = r"D:\face_recognition\images\img3.jpeg"
# Detect faces in the image
faces = detect_faces(filename)
# Draw bounding boxes around detected faces and count them
num_faces = draw_facebox(filename, faces)
print("Total number of faces found in the image:", num_faces)