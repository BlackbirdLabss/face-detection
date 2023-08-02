from flask import Response, Flask, render_template, jsonify, request
import threading
import argparse 
import datetime, time
import cv2
import face_recognition
import pdb
import os

outputFrameHOG = None
outputFrameHaarcascade = None
count_faces_hog = 0
count_faces_haarcascade = 0
data_face_hog = 0
data_face_haarcascade = 0

app = Flask(__name__)

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
source = 0
cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
time.sleep(2.0)

@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


def stream(frameCount):
    global outputFrameHOG, lock
    
    while True:
        grabbed, frame = cap.read()
        frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        with lock:
            outputFrameHOG = frame.copy()

def bounding_box_haarcascade(vid):
    global data_face_haarcascade, count_faces_haarcascade

    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    humans = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    if len(humans) == 0:
        data_face_haarcascade = 0
    else:
        data_face_haarcascade = len(humans)

    if count_faces_haarcascade < len(humans):
        count_faces_haarcascade = len(humans)
    for (x, y, w, h) in humans:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return humans

def haarcascade():
    total = 0
    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            continue
        else:


            faces = bounding_box_haarcascade(
               frame
            )
            (flag, encodedImage) = cv2.imencode(".jpg", frame)
 
            # ensure the frame was successfully encoded
            if not flag:
                continue
 
            # yield the output frame in the byte format
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                  bytearray(encodedImage) + b'\r\n')




def hog():
    global data_face_hog, count_faces_hog
    
    total = 0
    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            continue
        else:
            frame_resized = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            all_face_locations = face_recognition.face_locations(frame_resized,number_of_times_to_upsample=1,model='hog')
            #looping through the face locations
            if len(all_face_locations) == 0:
               data_face_hog = 0
               continue
            for index,current_face_location in enumerate(all_face_locations):
               #splitting the tuple to get the four position values of current face
               top_pos,right_pos,bottom_pos,left_pos = current_face_location
               #change the position maginitude to fit the actual size video frame
               top_pos = top_pos*4
               right_pos = right_pos*4
               bottom_pos = bottom_pos*4
               left_pos = left_pos*4
               data_face_hog = index+1
               if count_faces_hog < (index+1):
                  count_faces_hog = (index+1)
               #printing the location of current face
               #print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
               #draw rectangle around the face detected
               cv2.rectangle(frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", frame)
 
            # ensure the frame was successfully encoded
            if not flag:
                continue
 
            # yield the output frame in the byte format
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                  bytearray(encodedImage) + b'\r\n')

            
        

@app.route("/video_feed")
def video_feed():
    time.sleep(2)
    # return the response generated along with the specific media
    # type (mime type)
    return Response(hog(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed_second")
def video_feed_second():
    time.sleep(2)
    # return the response generated along with the specific media
    # type (mime type)
    return Response(haarcascade(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")
    
@app.route('/max/face/hog', methods= ['GET'])
def api_max_faces_hog():
    global count_faces_hog
    if(request.method == 'GET'):
        data = {
        "status": "success",
        "face_total": count_faces_hog
        }
    return jsonify(data)

@app.route('/max/face/haarcascade', methods= ['GET'])
def api_max_faces_haarcascade():
    global count_faces_haarcascade
    if(request.method == 'GET'):
        data = {
        "status": "success",
        "face_total": count_faces_haarcascade
        }
    return jsonify(data)

@app.route('/face/hog', methods= ['GET'])
def api_count_faces_hog():
    global data_face_hog
    if(request.method == 'GET'):
        data = {
        "status": "success",
        "face_total": data_face_hog
        }
    return jsonify(data)

@app.route('/face/haarcascade', methods= ['GET'])
def api_count_faces_haarcascade():
    global data_face_haarcascade
    if(request.method == 'GET'):
        data = {
        "status": "success",
        "face_total": data_face_haarcascade
        }
    return jsonify(data)

# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, default='0.0.0.0', required=True,
        help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, default=5000, required=True,
        help="ephemeral port number of the server (1024 to 65535)")
    args = vars(ap.parse_args())
 
    # start a thread that will perform motion detection
    # t = threading.Thread(target=detect_motion, args=(args["frame_count"],))
    # t.daemon = True
    # t.start()
 
    t = threading.Thread(target=hog)
    t.daemon = True
    t.start()
 
    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
        threaded=True, use_reloader=False)
 
# release the video stream pointer
cap.release()
cap.stop()