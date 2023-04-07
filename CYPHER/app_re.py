# # from flask import Flask, request, jsonify
# # from __future__ import division, print_function, absolute_import

# # import os
# # import tensorflow as tf
# # import keras.backend.tensorflow_backend as KTF

# # from timeit import time
# # import warnings
# # import argparse

# # import sys
# # import cv2
# # import numpy as np
# # import base64
# # import requests
# # import urllib
# # from urllib import parse
# # import json
# # import random
# # import time
# # from PIL import Image
# # from collections import Counter
# # import operator

# # from yolo_v3 import YOLO3
# # from yolo_v4 import YOLO4
# # from deep_sort import preprocessing
# # from deep_sort import nn_matching
# # from deep_sort.detection import Detection
# # from deep_sort.tracker import Tracker
# # from tools import generate_detections as gdet
# # from deep_sort.detection import Detection as ddet

# # from reid import REID
# # import copy

# # parser = argparse.ArgumentParser()
# # parser.add_argument('--version', help='Model(yolo_v3 or yolo_v4)', default='yolo_v4')
# # parser.add_argument('--videos', nargs='+', help='List of videos', required=True)
# # parser.add_argument('-all', help='Combine all videos into one', default=True)
# # args = parser.parse_args()  # vars(parser.parse_args())

# # app = Flask(__name__)

# # @app.route('/yolo_deepsort', methods=['POST'])
# # def yolo_deepsort():
# #     file = request.files['file']
# #     npimg = np.fromstring(file.read(), np.uint8)
# #     img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
# #     yolo = YOLO4()
# #     max_cosine_distance = 0.2
# #     nn_budget = None
# #     nms_max_overlap = 0.4
# #     model_filename = 'model_data/models/mars-small128.pb'
# #     encoder = gdet.create_box_encoder(model_filename, batch_size=1)
# #     metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
# #     tracker = Tracker(metric, max_age=100)

# #     boxs = yolo.detect_image(Image.fromarray(img))
# #     features = encoder(img, boxs)
# #     detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
# #     boxes = np.array([d.tlwh for d in detections])
# #     scores = np.array([d.confidence for d in detections])
# #     indices = preprocessing.delete_overlap_box(boxes, nms_max_overlap, scores)
# #     detections = [detections[i] for i in indices]

# #     tracker.predict()
# #     tracker.update(detections)
# #     ids_per_frame = []
# #     for track in tracker.tracks:
# #         if not track.is_confirmed() or track.time_since_update > 1:
# #             continue

# #         bbox = track.to_tlbr()
# #         area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
# #         ids_per_frame.append(track.track_id)
# #         cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
# #         cv2.putText(img, str(track.track_id), (int(bbox[0]), int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# #     return jsonify(ids_per_frame)

# # if __name__ == '__main__':
# #     app.run()




# from flask import Flask, render_template, request, jsonify, Response
# from __future__ import division, print_function, absolute_import
# import os
# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
# from timeit import time
# import warnings
# import argparse
# import sys
# import cv2
# import numpy as np
# import base64
# import requests
# import urllib
# from urllib import parse
# import json
# import random
# import time
# from PIL import Image
# from collections import Counter
# import operator

# from yolo_v3 import YOLO3
# from yolo_v4 import YOLO4
# from deep_sort import preprocessing
# from deep_sort import nn_matching
# from deep_sort.detection import Detection
# from deep_sort.tracker import Tracker
# from tools import generate_detections as gdet
# from deep_sort.detection import Detection as ddet

# from reid import REID
# import copy

# parser = argparse.ArgumentParser()
# parser.add_argument('--version', help='Model(yolo_v3 or yolo_v4)', default='yolo_v4')
# parser.add_argument('--videos', nargs='+', help='List of videos', required=True)
# parser.add_argument('-all', help='Combine all videos into one', default=True)
# args = parser.parse_args()  # vars(parser.parse_args())

# app = Flask(__name__)

# @app.route('/yolo_deepsort', methods=['POST'])
# def yolo_deepsort():
#     file = request.files['file']
#     npimg = np.fromstring(file.read(), np.uint8)
#     img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
#     yolo = YOLO4()
#     max_cosine_distance = 0.2
#     nn_budget = None
#     nms_max_overlap = 0.4
#     model_filename = 'model_data/models/mars-small128.pb'
#     encoder = gdet.create_box_encoder(model_filename, batch_size=1)
#     metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
#     tracker = Tracker(metric, max_age=100)

#     boxs = yolo.detect_image(Image.fromarray(img))
#     features = encoder(img, boxs)
#     detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
#     boxes = np.array([d.tlwh for d in detections])
#     scores = np.array([d.confidence for d in detections])
#     indices = preprocessing.delete_overlap_box(boxes, nms_max_overlap, scores)
#     detections = [detections[i] for i in indices]

#     tracker.predict()
#     tracker.update(detections)
#     ids_per_frame = []
#     for track in tracker.tracks:
#         if not track.is_confirmed() or track.time_since_update > 1:
#             continue

#         bbox = track.to_tlbr()
#         area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
#         ids_per_frame.append(track.track_id)
#         cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
#         cv2.putText(img, str(track.track_id), (int(bbox[0]), int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#     return jsonify(ids_per_frame)
# @app.route('/') 
# def index(): 
#     return render_template('main.html') 

# @app.route("/video_feed")
# def video_feed():
#     # to read from file
#     video_path = "weapon_01.mp4"
#     cap = cv2.VideoCapture(video_path)

#     yolo = YOLO4()
#     max_cosine_distance = 0.2
#     nn_budget = None
#     nms_max_overlap = 0.4
#     model_filename = 'model_data/models/mars-small128.pb'
#     encoder = gdet.create_box_encoder(model_filename, batch_size=1)
#     metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
#     tracker = Tracker(metric, max_age=100)

#     def generate_frames():
#         while True:
#             success, frame = cap.read()
#             if not success:
#                 break

#             boxs = yolo.detect_image(Image.fromarray(frame))
#             features = encoder(frame, boxs)
#             detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
#             boxes = np.array([d.tlwh for d in detections])
#             scores = np.array([d.confidence for d in detections])
#             indices = preprocessing.delete_overlap_box(boxes, nms_max_overlap, scores)
#             detections = [detections[i] for i in indices]

#             tracker.predict()
#             tracker.update(detections)
#             ids_per_frame = []
#             for track in tracker.tracks:
#                 if not track.is_confirmed() or track.time_since_update > 1:
#                     continue

#                 bbox = track.to_tlbr()
#                 area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
#                 ids_per_frame.append(track.track_id)
#                 cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
#                 cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#         cap.release()

#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# if __name__ == '__main__':
#     app.run(debug=True)




# from flask import Flask, render_template, request, jsonify, Response
# # from __future__ import division, print_function, absolute_import
# import os
# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
# from timeit import time
# import warnings
# import argparse
# import sys
# import cv2
# import numpy as np
# import base64
# import requests
# import urllib
# from urllib import parse
# import json
# import random
# import time
# from PIL import Image
# from collections import Counter
# import operator

# from yolo_v3 import YOLO3
# from yolo_v4 import YOLO4
# from deep_sort import preprocessing
# from deep_sort import nn_matching
# from deep_sort.detection import Detection
# from deep_sort.tracker import Tracker
# from tools import generate_detections as gdet
# from deep_sort.detection import Detection as ddet

# from reid import REID
# import copy

# parser = argparse.ArgumentParser()
# # parser.add_argument('--version', help='Model(yolo_v3 or yolo_v4)', default='yolo_v4')
# # parser.add_argument('--videos', nargs='+', help='List of videos', required=True)
# # parser.add_argument('-all', help='Combine all videos into one', default=True)

# parser.add_argument('--version', help='Model(yolo_v3 or yolo_v4)', default='yolo_v4')
# parser.add_argument('--video', help='Video file path', default='person.mp4')
# args = parser.parse_args()  # vars(parser.parse_args())

# app = Flask(__name__)

# @app.route('/yolo_deepsort', methods=['POST'])
# def yolo_deepsort():
#     file = request.files['file']
#     npimg = np.fromstring(file.read(), np.uint8)
#     img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
#     yolo = YOLO4()
#     max_cosine_distance = 0.2
#     nn_budget = None
#     nms_max_overlap = 0.4
#     model_filename = 'model_data/models/mars-small128.pb'
#     encoder = gdet.create_box_encoder(model_filename, batch_size=1)
#     metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
#     tracker = Tracker(metric, max_age=100)

#     boxs = yolo.detect_image(Image.fromarray(img))
#     features = encoder(img, boxs)
#     detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
#     boxes = np.array([d.tlwh for d in detections])
#     scores = np.array([d.confidence for d in detections])
#     indices = preprocessing.delete_overlap_box(boxes, nms_max_overlap, scores)
#     detections = [detections[i] for i in indices]

#     tracker.predict()
#     tracker.update(detections)
#     ids_per_frame = []
#     for track in tracker.tracks:
#         if not track.is_confirmed() or track.time_since_update > 1:
#             continue

#         bbox = track.to_tlbr()
#         area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
#         ids_per_frame.append(track.track_id)
#         cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
#         cv2.putText(img, str(track.track_id), (int(bbox[0]), int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#     return jsonify(ids_per_frame)
# @app.route('/') 
# def index(): 
#     return render_template('main.html') 

# @app.route("/video_feed")
# def video_feed():
#     # to read from file
#     video_path = "person.mp4"
#     cap = cv2.VideoCapture(video_path)

#     yolo = YOLO4()
#     max_cosine_distance = 0.2
#     nn_budget = None
#     nms_max_overlap = 0.4
#     model_filename = 'model_data/models/mars-small128.pb'
#     encoder = gdet.create_box_encoder(model_filename, batch_size=1)
#     metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
#     tracker = Tracker(metric, max_age=100)

#     def generate_frames():
#         while True:
#             success, frame = cap.read()
#             if not success:
#                 break

#             boxs = yolo.detect_image(Image.fromarray(frame))
#             features = encoder(frame, boxs)
#             detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
#             boxes = np.array([d.tlwh for d in detections])
#             scores = np.array([d.confidence for d in detections])
#             indices = preprocessing.delete_overlap_box(boxes, nms_max_overlap, scores)
#             detections = [detections[i] for i in indices]

#             tracker.predict()
#             tracker.update(detections)
#             ids_per_frame = []
#             for track in tracker.tracks:
#                 if not track.is_confirmed() or track.time_since_update > 1:
#                     continue

#                 bbox = track.to_tlbr()
#                 area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
#                 ids_per_frame.append(track.track_id)
#                 cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
#                 cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#         cap.release()

#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# if __name__ == '__main__':
#     app.run(debug=True)







# from flask import Flask, render_template, request, jsonify, Response
# import os
# import tensorflow as tf
# from tensorflow import keras
# # import tensorflow.keras.backend as KTF
# import keras.layers.advanced_activations as activations
# from tensorflow.keras import backend as KTF
# from timeit import time
# import warnings
# import argparse
# import sys
# import cv2
# import numpy as np
# import base64
# import requests
# import urllib
# from urllib import parse
# import json
# import random
# import time
# from PIL import Image
# from collections import Counter
# import operator


# LASTTTTTTTTTTTTTTTT

# from flask import Flask, render_template, request, jsonify, Response
# # from __future__ import division, print_function, absolute_import
# import os
# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
# from timeit import time
# import warnings
# import argparse
# import sys
# import cv2
# import numpy as np
# import base64
# import requests
# import urllib
# from urllib import parse
# import json
# import random
# import time
# from PIL import Image
# from collections import Counter
# import operator

# from yolo_v3 import YOLO3
# from yolo_v4 import YOLO4
# from deep_sort import preprocessing
# from deep_sort import nn_matching
# from deep_sort.detection import Detection
# from deep_sort.tracker import Tracker
# from tools import generate_detections as gdet
# from deep_sort.detection import Detection as ddet

# from reid import REID
# import copy

# parser = argparse.ArgumentParser()
# parser.add_argument('--version', help='Model(yolo_v3 or yolo_v4)', default='yolo_v4')
# parser.add_argument('--video', help='Video file path', default='person.mp4')
# args = parser.parse_args()

# app = Flask(__name__)

# @app.route('/yolo_deepsort', methods=['POST'])
# def yolo_deepsort():
#     file = request.files['file']
#     npimg = np.fromstring(file.read(), np.uint8)
#     img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
#     yolo = YOLO4()
#     max_cosine_distance = 0.2
#     nn_budget = None
#     nms_max_overlap = 0.4
#     model_filename = 'model_data/models/mars-small128.pb'
#     encoder = gdet.create_box_encoder(model_filename, batch_size=1)
#     metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
#     tracker = Tracker(metric, max_age=100)

#     boxs = yolo.detect_image(Image.fromarray(img))
#     features = encoder(img, boxs)
#     detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
#     boxes = np.array([d.tlwh for d in detections])
#     scores = np.array([d.confidence for d in detections])
#     indices = preprocessing.delete_overlap_box(boxes, nms_max_overlap, scores)
#     detections = [detections[i] for i in indices]

#     tracker.predict()
#     tracker.update(detections)
#     ids_per_frame = []
#     for track in tracker.tracks:
#         if not track.is_confirmed() or track.time_since_update > 1:
#             continue

#         bbox = track.to_tlbr()
#         area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
#         ids_per_frame.append(track.track_id)
#         cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
#         cv2.putText(img, str(track.track_id), (int(bbox[0]), int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#     return jsonify(ids_per_frame)

# @app.route('/') 
# def index(): 
#     return render_template('main.html')

# @app.route("/video_feed")
# def video_feed():
#     # to read from file
#     video_path = "person.mp4"
#     cap = cv2.VideoCapture(video_path)

#     yolo = YOLO4()
#     max_cosine_distance = 0.2
#     nn_budget = None
#     nms_max_overlap = 0.4
#     model_filename = 'model_data/models/mars-small128.pb'
#     encoder = gdet.create_box_encoder(model_filename, batch_size=1)
#     metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
#     tracker = Tracker(metric, max_age=100)

#     def generate_frames():
#         while True:
#             success, frame = cap.read()
#             if not success:
#                 break

#             boxs = yolo.detect_image(Image.fromarray(frame))
#             features = encoder(frame, boxs)
#             detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
#             boxes = np.array([d.tlwh for d in detections])
#             scores = np.array([d.confidence for d in detections])
#             indices = preprocessing.delete_overlap_box(boxes, nms_max_overlap, scores)
#             detections = [detections[i] for i in indices]

#             tracker.predict()
#             tracker.update(detections)
#             ids_per_frame = []
#             for track in tracker.tracks:
#                 if not track.is_confirmed() or track.time_since_update > 1:
#                     continue

#                 bbox = track.to_tlbr()
#                 area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
#                 ids_per_frame.append(track.track_id)
#                 cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
#                 cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#         cap.release()

#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# if __name__ == '__main__':
#     app.run(debug=True)


# FOR TF 2.11.0 AND KERAS 2.11.0
from flask import Flask, render_template, request, jsonify, Response
import os
import tensorflow as tf
import tensorflow.keras.backend as ktf
from timeit import time
import warnings
import argparse
import sys
import cv2
import numpy as np
import base64
import requests
import urllib
from urllib import parse
import json
import random
import time
from PIL import Image
from collections import Counter
import operator

from yolo_v3 import YOLO3
from yolo_v4 import YOLO4
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

from reid import REID
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--version', help='Model(yolo_v3 or yolo_v4)', default='yolo_v4')
parser.add_argument('--video', help='Video file path', default='person.mp4')
args = parser.parse_args()

app = Flask(__name__)

@app.route('/yolo_deepsort', methods=['POST'])
def yolo_deepsort():
    file = request.files['file']
    npimg = np.fromstring(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    yolo = YOLO4()
    max_cosine_distance = 0.2
    nn_budget = None
    nms_max_overlap = 0.4
    model_filename = 'model_data/models/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age=100)

    boxs = yolo.detect_image(Image.fromarray(img))
    features = encoder(img, boxs)
    detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = preprocessing.delete_overlap_box(boxes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    tracker.predict()
    tracker.update(detections)
    ids_per_frame = []
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        bbox = track.to_tlbr()
        area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
        ids_per_frame.append(track.track_id)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        cv2.putText(img, str(track.track_id), (int(bbox[0]), int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return jsonify(ids_per_frame)

@app.route('/') 
def index(): 
    return render_template('main.html')

@app.route("/video_feed")
def video_feed():
    # to read from file
    video_path = "person.mp4"
    cap = cv2.VideoCapture(video_path)

    yolo = YOLO4()
    max_cosine_distance = 0.2
    nn_budget = None
    nms_max_overlap = 0.4
    model_filename = 'model_data/models/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age=100)

    def generate_frames():
        while True:
            success, frame = cap.read()
            if not success:
                break

            boxs = yolo.detect_image(Image.fromarray(frame))
            features = encoder(frame, boxs)
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.delete_overlap_box(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            tracker.predict()
            tracker.update(detections)
            ids_per_frame = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue



                bbox = track.to_tlbr()
                area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
                ids_per_frame.append(track.track_id)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Convert the frame to JPEG format
            frame = cv2.imencode('.jpg', frame)[1].tobytes()

            # Yield the JPEG frame as a response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

