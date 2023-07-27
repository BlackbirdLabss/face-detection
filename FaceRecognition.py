import face_recognition
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from gevent import monkey 
from VideoCapture import VideoCap
from gevent.pywsgi import WSGIServer
from flask import Flask, render_template, Response, jsonify, request

app = Flask(__name__)
resolution_x = 0.25
resolution_y = 0.25


#rtsp://admin:FMOSMJ@192.168.221.199:554/Streaming/Channels/102
webcam = VideoCap(0, resolution_x, resolution_y)


def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    humans = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    webcam.count_faces_haarcascade.append(len(humans))
    #print("human {}".format(len(faces)))
    for (x, y, w, h) in humans:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return humans


def gen_frames_second():
    while True:        
        webcam._update_current_frame()
        result, video_frame_second = webcam.get_current_frame_read()  # read frames from the video
        if result is False:
            break  # terminate the loop if the frame is not read successfully
        else:
            faces = detect_bounding_box(
                video_frame_second
            )
            ret, buffer = cv2.imencode('.jpg', video_frame_second)
            #cv2.putText(buffer, "Total Human: "+str(sum_head), (10, 50), 0, 5e-3 * 130, (0,0,255), 2)
            buffer_frame_second = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer_frame_second + b'\r\n')  # concat frame one by one and show result
        
def gen_frames():  # generate frame by frame from camera
    while True:    
        webcam._update_current_frame()
        webcam._resize_current_frame(resolution_x, resolution_y)
        tm = cv2.TickMeter()
        tm.start()
        webcam.face_recog(1, "hog")
        tm.stop()

        all_face_locations = webcam.get_all_face_locations()

        for index, current_face_location in enumerate(all_face_locations):
            # splitting the tuple to get the four position values of current face
            top_pos, right_pos, bottom_pos, left_pos = current_face_location
            # change the position magnitude to fit the actual size video frame
            top_pos = top_pos*int(1/resolution_y)
            right_pos = right_pos*int(1/resolution_x)
            bottom_pos = bottom_pos*int(1/resolution_y)
            left_pos = left_pos*int(1/resolution_x)
            # printing the location of current face
            #print('Found face {} at cordinate top:{}, right:{}, bottom:{}, left:{}'.format(index+1, top_pos, right_pos, bottom_pos, left_pos))
            webcam.count_faces_hog.append(index+1)
            #print('Found face {} '.format(index+1))
            # draw rectangle around the face detected
            rect_image = cv2.rectangle(webcam.get_current_frame(), (left_pos, top_pos), (right_pos, bottom_pos), (0,0,255), 2)
            cv2.putText(rect_image, "Manusia", (left_pos, top_pos-8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            cv2.putText(webcam.get_current_frame(), 'FPS: {:.2f}'.format(tm.getFPS()), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        ret, buffer = cv2.imencode('.jpg', webcam.get_current_frame())
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
    

@app.route('/input/source/', methods= ['POST'])
def input_source():
    input_value = request.form["input_value"]
    fx = request.form["resolution"]
    fy = request.form["resolution"]

    if(input_value.isdigit()):
        input_value = int(input_value)
    
    fx = float(fx)
    fy = float(fy)

    webcam.input_source(input_value, fx, fy)
    webcam._update_current_frame()

    if(request.method == 'POST'):
        data = {
        "status" : "success",
        "input_source" : input_value
        }
        
    return jsonify(data)

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_second')
def video_feed_second():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames_second(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/face/hog', methods= ['GET'])
def api_count_faces_hog():
    if(request.method == 'GET'):
        data = {
        "status": "success",
        "face_total": webcam.get_count_faces_hog()
        }
    return jsonify(data)

@app.route('/face/haarcascade', methods= ['GET'])
def api_count_faces_haarcascade():
    if(request.method == 'GET'):
        data = {
        "status": "success",
        "face_total": webcam.get_count_faces_haarcascade()
        }
    return jsonify(data)

@app.route('/check/camera', methods= ['GET'])
def check_camera():
    if(request.method == 'GET'):    
        valid_cams = []
        for i in range(8):
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
    print(webcam.get_count_face)

    

    