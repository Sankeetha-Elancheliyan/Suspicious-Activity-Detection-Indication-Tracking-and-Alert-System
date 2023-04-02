import cv2
import numpy as np
import os
import winsound

from fastapi import FastAPI, Response

app = FastAPI()

# initialize the camera and object detection model
camera = cv2.VideoCapture(0)
classNames = []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
thres = 0.5 # detection threshold

freq = 1000
dur = 100

def detect_objects(frame):
    # perform object detection on the frame
    classIds, confs, bbox = net.detect(frame, confThreshold=thres)
    # draw bounding boxes and labels for detected persons
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId-1]
            if className == 'person':
                # play the alert sound
                winsound.Beep(freq, dur)
                cv2.rectangle(frame, box, color=(0, 0, 255), thickness=2)
                cv2.putText(frame, className, (box[0]+10, box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, str(round(confidence*100, 2)), (box[0]+200, box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    # encode the frame as JPEG and return it
    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    return frame

@app.get('/')
async def index():
    return 'Welcome to the camera feed!'

@app.get('/video_feed')
async def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    while True:
        # read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            # perform object detection on the frame
            frame = detect_objects(frame)
            # yield the frame in the response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
