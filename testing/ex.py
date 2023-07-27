from flask import Response, Flask, render_template
import threading
import argparse 
import datetime, time
import cv2
import face_recognition
outputFrame = None
lock = threading.Lock()

app = Flask(__name__)

source = 'rtsp://admin:FMOSMJ@192.168.221.199:554'
cap = cv2.VideoCapture(source)
time.sleep(2.0)

@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


def stream(frameCount):
    global outputFrame, lock
    
    while True:
        ret_val, frame = cap.read()
        frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        with lock:
            outputFrame = frame.copy()

def face_recog():
    global outputFrame, lock
    
    all_face_locations = []
    total = 0
    while True:
        ret_val, frame = cap.read()
        if not ret_val:
            continue
        else:
            frame_resized = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            if total > 32:
                all_face_locations = face_recognition.face_locations(frame_resized,number_of_times_to_upsample=1,model='hog')
                #looping through the face locations
                for index,current_face_location in enumerate(all_face_locations):
                    #splitting the tuple to get the four position values of current face
                    top_pos,right_pos,bottom_pos,left_pos = current_face_location
                    #change the position maginitude to fit the actual size video frame
                    top_pos = top_pos*4
                    right_pos = right_pos*4
                    bottom_pos = bottom_pos*4
                    left_pos = left_pos*4
                    #printing the location of current face
                    #print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
                    #draw rectangle around the face detected
                    cv2.rectangle(frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
                
            total+=1
            
            with lock:
                outputFrame = frame.copy()
            
        
def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
 
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
 
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
 
            # ensure the frame was successfully encoded
            if not flag:
                continue
 
        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")
    
    
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

    t = threading.Thread(target=face_recog)
    t.daemon = True
    t.start()
 
    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
        threaded=True, use_reloader=False)
 
# release the video stream pointer
cap.release()
cap.stop()