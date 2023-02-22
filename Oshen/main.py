import cv2
import winsound
import datetime

freq = 1000
dur = 100
thres = 0.5 # Threshold to detect objects

cap1 = cv2.VideoCapture(0)  # first camera
cap2 = cv2.VideoCapture(1)  # second camera
cap3 = cv2.VideoCapture(2)  # third camera
cap4 = cv2.VideoCapture(3)  # fourth camera

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
now = datetime.datetime.now()
date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

output_file1 = f"output_cam1_{date_time}.mp4"
output_file2 = f"output_cam2_{date_time}.mp4"
output_file3 = f"output_cam3_{date_time}.mp4"
output_file4 = f"output_cam4_{date_time}.mp4"

out1 = cv2.VideoWriter(output_file1,fourcc,5.0,(640, 480))
out2 = cv2.VideoWriter(output_file2,fourcc,5.0,(640, 480))
out3 = cv2.VideoWriter(output_file3,fourcc,5.0,(640, 480))
out4 = cv2.VideoWriter(output_file4,fourcc,5.0,(640, 480))

cap1.set(3,640)
cap1.set(4,480)

cap2.set(3,640)
cap2.set(4,480)

cap3.set(3,640)
cap3.set(4,480)

cap4.set(3,640)
cap4.set(4,480)

classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    # read frames from both cameras
    success1, img1 = cap1.read()
    success2, img2 = cap2.read()
    success3, img3 = cap3.read()
    success4, img4 = cap4.read()

    # get the current date and time
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    current_date = now.strftime("%Y-%m-%d")

    # add the current date and time to the frames
    cv2.putText(img1, current_date + " " + current_time, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img2, current_date + " " + current_time, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img3, current_date + " " + current_time, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img4, current_date + " " + current_time, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # detect objects in the frames from camera 1
    classIds1, confs1, bbox1 = net.detect(img1,confThreshold=thres)
    if len(classIds1) != 0:
        for classId, confience,box in zip(classIds1.flatten(),confs1.flatten(),bbox1):
            cv2.rectangle(img1,box,color=(0,0,255),thickness=2)
            if classId == [1]:
                out1.write(img1)
                winsound.Beep(freq, dur)
            cv2.putText(img1,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            cv2.putText(img1,str(round(confience*100,2)),(box[0]+200,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

    # detect objects in the frames from camera 2
    classIds2, confs2, bbox2 = net.detect(img2, confThreshold=thres)
    if len(classIds2) != 0:
        for classId, confience, box in zip(classIds2.flatten(), confs2.flatten(), bbox2):
            cv2.rectangle(img2, box, color=(0, 0, 255), thickness=2)
            if classId == [1]:
                out2.write(img2)
                winsound.Beep(freq, dur)
                cv2.putText(img2, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(img2, str(round(confience * 100, 2)), (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    # detect objects in the frames from camera 3
    classIds3, confs3, bbox3 = net.detect(img3, confThreshold=thres)
    if len(classIds3) != 0:
        for classId, confience, box in zip(classIds1.flatten(), confs1.flatten(), bbox1):
            cv2.rectangle(img3, box, color=(0, 0, 255), thickness=2)
            if classId == [1]:
                out3.write(img3)
                winsound.Beep(freq, dur)
                cv2.putText(img3, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(img3, str(round(confience * 100, 2)), (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    # detect objects in the frames from camera 4
    classIds4, confs4, bbox4 = net.detect(img4, confThreshold=thres)
    if len(classIds4) != 0:
         for classId, confience, box in zip(classIds1.flatten(), confs1.flatten(), bbox1):
            cv2.rectangle(img4, box, color=(0, 0, 255), thickness=2)
            if classId == [1]:
                out4.write(img4)
                winsound.Beep(freq, dur)
                cv2.putText(img4, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(img4, str(round(confience * 100, 2)), (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)


    cv2.imshow("Output0", img1)
    cv2.imshow("Output1", img2)
    cv2.imshow("Output2", img3)
    cv2.imshow("Output3", img4)

    cv2.waitKey(1)