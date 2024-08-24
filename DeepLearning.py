import jetson.inference
import jetson.utils
import cv2
import numpy as np
import time

# Video dimensions
width = 1280
height = 720
dispW = width
dispH = height
flip = 2

# Open video file (use camera device path for live feed)
cam1 = cv2.VideoCapture("/home/priya/Documents/stem.mp4")

# Initialize the classification model
net = jetson.inference.imageNet(
    'resnet18',
    [
        '--model=/home/priya/Downloads/jetson-inference/python/training/classification/myModel/resnet18.onnx',
        '--input_blob=input_0',
        '--output_blob=output_0',
        '--labels=/home/priya/Downloads/jetson-inference/mytrain/labels.txt'
    ]
)

font = cv2.FONT_HERSHEY_SIMPLEX
timeMark = time.time()
fpsFilter = 0

while True:
    # Read frame from video
    _, frame = cam1.read()
    
    # Convert frame and perform classification
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA).astype(np.float32)
    img = jetson.utils.cudaFromNumpy(img)
    classID, confidence = net.Classify(img, width, height)
    
    # Get class description and compute FPS
    item = net.GetClassDesc(classID)
    dt = time.time() - timeMark
    fps = 1 / dt
    fpsFilter = 0.95 * fpsFilter + 0.05 * fps
    timeMark = time.time()
    
    # Display FPS and classification result
    cv2.putText(frame, f'{round(fpsFilter, 1)} fps {item}', (0, 30), font, 1, (0, 0, 255), 2)
    cv2.imshow('recCam', frame)
    cv2.moveWindow('recCam', 0, 0)
    
    # Exit on 'q' key
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cam1.release()
cv2.destroyAllWindows()
