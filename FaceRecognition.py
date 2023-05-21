import face_recognition
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np
from flask import Flask, render_template, Response
from gevent.pywsgi import WSGIServer
from gevent import monkey

app = Flask(__name__)    
from VideoCapture import VideoCap
resolution_x = 0.25
resolution_y = 0.25
webcam = VideoCap(0, resolution_x, resolution_y)
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

fpss = 0 

all_face_locations = []

def gen_frames_second():
    while True:
        result, video_frame = webcam.get_current_frame_read()  # read frames from the video
        if result is False:
            break  # terminate the loop if the frame is not read successfully
        else:
            webcam._update_current_frame()
            webcam._resize_current_frame(resolution_x, resolution_y)
            faces = detect_bounding_box(
                video_frame
            )
            ret, buffer = cv2.imencode('.jpg', video_frame)
            frame_second = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_second + b'\r\n')  # concat frame one by one and show result
        
def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = webcam.get_current_frame_read()  # read the camera frame
        if not success:
            break
        else:    
            webcam._update_current_frame()
            webcam._resize_current_frame(resolution_x, resolution_y)
            webcam.get_frame_enhancement(10,2)
            tm = cv2.TickMeter()
            tm.start()
            webcam.face_recog(2, "hog")
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
                print('Found face {} at cordinate top:{}, right:{}, bottom:{}, left:{}'.format(index+1, top_pos, right_pos, bottom_pos, left_pos))
                # draw rectangle around the face detected
                rect_image = cv2.rectangle(webcam.get_current_frame(), (left_pos, top_pos), (right_pos, bottom_pos), (0,0,255), 2)
                cv2.putText(rect_image, "Manusia ", (left_pos, top_pos-8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                cv2.putText(webcam.get_current_frame(), 'FPS: {:.2f}'.format(tm.getFPS()), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            
            ret, buffer = cv2.imencode('.jpg', webcam.get_current_frame())
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_second')
def video_feed_second():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames_second(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def main():

    # use gevent WSGI server instead of the Flask
    # instead of 5000, you can define whatever port you want.
    http = WSGIServer(('', 5000), app.wsgi_app) 

    # Serve your application
    http.serve_forever()

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)

    

    