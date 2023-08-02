import face_recognition
import cv2
import numpy as np


class VideoStream:
    def __init__(self, source):
        all_face_locations = []
        self.count_faces_hog = 0
        self.count_faces_haarcascade = 0
        self.data_face_hog = 0
        self.data_face_haarcascade = 0
        self.video_capture = cv2.VideoCapture(source)
    
    def update_frame(self):
        (self.grabbed, self.frame) = self.video_capture.read()

        return (self.grabbed, self.frame)
    
    def resize_frame(self, frame, resolution_x, resolution_y):
        self.frame_resized = cv2.resize(frame, (0, 0), fx=resolution_x, fy=resolution_y)
        return self.frame_resized

    def face_recog(self, upsample, model):
        self.all_face_locations = face_recognition.face_locations(
            self.frame_resized, upsample, model)
        return self.all_face_locations
    
    def get_frame(self):
        return self.frame
    
    def input_source(self, source, resolution_x, resolution_y):
        self.video_capture = cv2.VideoCapture(source)
        self.ret, self.frame = self.video_capture.read()
        self.frame_resized = cv2.resize(self.frame, (0, 0), fx=resolution_x, fy=resolution_y)

    def get_count_faces_haarcascade(self):
        max_face_haarcascade = max(self.count_faces_haarcascade)
        return max_face_haarcascade

    def get_count_faces_hog(self):
        max_face_hog = max(self.count_faces_hog)
        return max_face_hog
    
    

    
