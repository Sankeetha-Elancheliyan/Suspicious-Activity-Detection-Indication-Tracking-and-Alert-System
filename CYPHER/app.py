# VIOLENCE,ROBBERY & INTRUSION,WEAPON

from flask import Flask, render_template, Response
import cv2
from collections import deque
import numpy as np
import tensorflow as tf
from keras.models import load_model
from ultralytics import YOLO
import cv2
import imutils
import threading

app = Flask(__name__)
lock = threading.Lock()


# Load the pre-trained models
Robbery_model = load_model('r_model.h5')
Violence_model = load_model('model.h5')

# Constants
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 112
CLASSES_LIST_ROBBERY = ['Robbery', 'Normal']
CLASSES_LIST_VIOLENCE = ['Non-Violence', 'Violence']
SEQUENCE_LENGTH = 16

# Weapon detection model
weapon_model = YOLO('./weights/best.pt')

# Intrusion detection model
classNames = [] 
classFile = 'coco.names' 
with open(classFile,'rt') as f: 
    classNames = f.read().rstrip('\n').split('\n') 
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt' 
weightsPath = 'frozen_inference_graph.pb' 
intrusion_model = cv2.dnn_DetectionModel(weightsPath, configPath) 
intrusion_model.setInputSize(320, 320) 
intrusion_model.setInputScale(1.0/ 127.5) 
intrusion_model.setInputMean((127.5, 127.5, 127.5)) 
intrusion_model.setInputSwapRB(True) 
intrusion_thres = 0.5 # detection threshold 

# Define a deque to store video frames
frames_queue = deque(maxlen=SEQUENCE_LENGTH)

# Function to get the predicted class name and result string for robbery model
def get_prediction_robbery():
    # Pass the normalized frames to the model and get the predicted probabilities
    predicted_labels_probabilities = Robbery_model.predict(np.expand_dims(frames_queue, axis=0))[0]

    # Get the index of class with highest probability
    predicted_label = np.argmax(predicted_labels_probabilities)

    # Get the class name using the retrieved index
    predicted_class_name = CLASSES_LIST_ROBBERY[predicted_label]

    # Create the result string
    if predicted_class_name == "Robbery":
        result = f'{predicted_class_name}: {predicted_labels_probabilities[predicted_label]*100}'
    else:
        result = f'{predicted_class_name}: {100 - predicted_labels_probabilities[predicted_label]*100}'

    return predicted_class_name, result

# Function to get the predicted class name and result string for violence model
def get_prediction_violence():
    # Pass the normalized frames to the model and get the predicted probabilities
    predicted_labels_probabilities = Violence_model.predict(np.expand_dims(frames_queue, axis=0))[0]

    # Get the index of class with highest probability
    predicted_label = np.argmax(predicted_labels_probabilities)

    # Get the class name using the retrieved index
    predicted_class_name = CLASSES_LIST_VIOLENCE[predicted_label]

    # Create the result string
    if predicted_class_name == "Violence":
        result = f'{predicted_class_name}: {predicted_labels_probabilities[predicted_label]*100}'
    else:
        result = f'{predicted_class_name}: {100 - predicted_labels_probabilities[predicted_label]*100}'

    return predicted_class_name, result

# Define functions for detecting objects in frames

def detect_weapons(frame):
    result = weapon_model(frame)
    res_plotted = result[0].plot()
    return res_plotted

def detect_intrusions(frame):
    # perform object detection on the frame
    classIds, confs, bbox = intrusion_model.detect(frame, confThreshold=intrusion_thres)
    # draw bounding boxes and labels for detected persons
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId-1]
            if className == 'person':
                cv2.rectangle(frame, box, color=(0, 0, 255), thickness=2)
                cv2.putText(frame, className, (box[0]+10, box[1]+30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, str(round(confidence*100, 2)), (box[0]+200, box[1]+30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    return frame

def generate_frames(video_path=None, cam_device=0):
    # grab global references to the output frame and lock variables
    global outputFrame, lock
    # initialize the video stream
    if video_path is not None:
        vs = cv2.VideoCapture(video_path)
    else:
        vs = cv2.VideoCapture(cam_device)
    # loop over frames from the video stream
    while True:
        # read the next frame from the video stream
        (grabbed, frame) = vs.read()
        # if the frame was not grabbed, then we have reached the end of the video
        if not grabbed:
            break
        # resize the frame to have a maximum width of 400 pixels
        # frame = imutils.resize(frame, width=400)
        # apply some image processing if necessary

        # feed the frames to the object detection models like below
        frame = detect_weapons(frame)
        frame = detect_intrusions(frame)

        # ...
        # wait until the lock is acquired
        with lock:
            # update the output frame
            outputFrame = frame.copy()
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
    # release the video stream
    vs.release()
    

#Route for the video feed
@app.route('/video_feed')
def video_feed():
    # Specify the path of the video file
    video_path = 'weapon_01.mp4'

    def generate(video_path):
        # Create a VideoCapture object to read the video file
        video_reader = cv2.VideoCapture(video_path)

        while True:
            # Read a frame from the video feed
            ok, frame = video_reader.read()
            if not ok:
                break

            # Resize the frame to fixed dimensions
            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

            # Normalize the resized frame
            normalized_frame = resized_frame / 255

            # Append the pre-processed frame into the frames queue
            frames_queue.append(normalized_frame)

            # Check if the number of frames in the queue are equal to the fixed sequence length
            if len(frames_queue) == SEQUENCE_LENGTH:
                # Get the predicted class name and result string for Robbery model
                predicted_class_name_robbery, result_robbery = get_prediction_robbery()

                # Get the predicted class name and result string for Violence model
                predicted_class_name_violence, result_violence = get_prediction_violence()

                # Write the predicted class name and result for Robbery model on top of the frame
                if predicted_class_name_robbery == "Robbery":
                    cv2.putText(frame, result_robbery, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    cv2.putText(frame, result_robbery, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                # Write the predicted class name and result for Violence model on top of the frame
                if predicted_class_name_violence == "Violence":
                    cv2.putText(frame, result_violence, (5, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    cv2.putText(frame, result_violence, (5, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # Encode the frame as JPEG
            _, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()

            # Yield the encoded frame to be sent to the client
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video_feed_2")
def video_feed_2():
    # to read from file
    video_path = "weapon_02.mp4"
    return Response(generate_frames(video_path), mimetype="multipart/x-mixed-replace;boundary=frame")

# Main route that renders the HTML template
@app.route('/')
def index():
    return render_template('loginscreen.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/login')
def login():
    return render_template('loginscreen.html')

@app.route('/main')
def main():
    return render_template('main.html')

if __name__ == '__main__':
    app.run(debug=True)
