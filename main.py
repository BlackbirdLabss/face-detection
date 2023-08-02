import face_recognition
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from gevent import monkey 
from VideoStream import VideoStream
from gevent.pywsgi import WSGIServer
from flask import Flask, render_template, Response, jsonify, request

app = Flask(__name__)
resolution_x = 0.25
resolution_y = 0.25
change_input = False

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
source = 0
cap = VideoStream(source)


def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    frame_detect = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    if len(frame_detect) == 0:
        cap.data_face_haarcascade = 0
    else:
        cap.data_face_haarcascade = len(frame_detect)

    if cap.count_faces_haarcascade < len(frame_detect):
        cap.count_faces_haarcascade = len(frame_detect)
    for (x, y, w, h) in frame_detect:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return frame_detect

def gen_frames_second():
    global change_input
    change_input = True
    while True:      
        ret, frame = cap.update_frame()

        if not ret:
            continue

        faces = detect_bounding_box(
            frame
        )
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        buffer_frame_second = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer_frame_second + b'\r\n')
        if not change_input:
            break  

def gen_frames():
    global change_input
    change_input = False
    while True:           
        ret, frame = cap.update_frame()

        if not ret:
            continue

        cap.resize_frame(frame, resolution_x, resolution_y)
        tm = cv2.TickMeter()
        tm.start()
        all_face_locations = cap.face_recog(1, 'hog')
        tm.stop()
        
        #looping through the face locations
        if len(all_face_locations) == 0:
            cap.data_face_hog = 0
            continue

        for index, current_face_location in enumerate(all_face_locations):
            top_pos, right_pos, bottom_pos, left_pos = current_face_location
            top_pos = top_pos*int(1/resolution_y)
            right_pos = right_pos*int(1/resolution_x)
            bottom_pos = bottom_pos*int(1/resolution_y)
            left_pos = left_pos*int(1/resolution_x)

            cap.data_face_hog = index+1
            if cap.count_faces_hog < (index+1):
                cap.count_faces_hog = (index+1)

            rect_image = cv2.rectangle(cap.frame, (left_pos, top_pos), (right_pos, bottom_pos), (0,0,255), 2)
            cv2.putText(rect_image, "Manusia", (left_pos, top_pos-8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            cv2.putText(cap.frame, 'FPS: {:.2f}'.format(tm.getFPS()), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        ret, buffer = cv2.imencode('.jpg', cap.frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
               
        if change_input:
            break 

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_second')
def video_feed_second():
    return Response(gen_frames_second(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/input/source/', methods= ['POST'])
def input_source():
    input_value = request.form["input_value"]
    fx = request.form["resolution"]
    fy = request.form["resolution"]

    if(input_value.isdigit()):
        input_value = int(input_value)
    
    fx = float(fx)
    fy = float(fy)

    cap.input_source(input_value, fx, fy)
    cap.update_frame()

    if(request.method == 'POST'):
        data = {
        "status" : "success",
        "input_source" : input_value
        }
        
    return jsonify(data)


@app.route('/max/face/hog', methods= ['GET'])
def api_max_faces_hog():
    if(request.method == 'GET'):
        data = {
        "status": "success",
        "face_total": cap.count_faces_hog
        }
    return jsonify(data)

@app.route('/max/face/haarcascade', methods= ['GET'])
def api_max_faces_haarcascade():
    if(request.method == 'GET'):
        data = {
        "status": "success",
        "face_total": cap.count_faces_haarcascade
        }
    return jsonify(data)

@app.route('/face/hog', methods= ['GET'])
def api_count_faces_hog():
    if(request.method == 'GET'):
        data = {
        "status": "success",
        "face_total": cap.data_face_hog
        }
    return jsonify(data)

@app.route('/face/haarcascade', methods= ['GET'])
def api_count_faces_haarcascade():
    if(request.method == 'GET'):
        data = {
        "status": "success",
        "face_total": cap.data_face_haarcascade
        }
    return jsonify(data)

@app.route('/check/camera', methods= ['GET'])
def check_camera():
    if(request.method == 'GET'):    
        valid_cams = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap is None or not cap.isOpened():
                print('Warning: unable to open video source: ', i)
            else:
                valid_cams.append(i)
        data = {
            "status" : "success",
            "video_source" :  valid_cams
        }
    return jsonify(data)

def main():

    # use gevent WSGI server instead of the Flask
    # instead of 5000, you can define whatever port you want.
    http = WSGIServer(('', 5000), app.wsgi_app) 

    # Serve your application
    http.serve_forever()

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, debug=True)