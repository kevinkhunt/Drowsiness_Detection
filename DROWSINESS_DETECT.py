import cv2
import dlib
import numpy as np
from math import hypot
import threading
import pygame



def start_sound():
    pygame.mixer.init()
    pygame.mixer.music.load('z.ogg')
    pygame.mixer.music.play()

def stop_sound():
    pygame.mixer.init()
    pygame.mixer.music.stop()


def midpoint(p1,p2):
    return int((p1.x+p2.x)/2) ,int((p1.y+p2.y)/2)


def get_blinking_ratio(eye_points,facial_landmarks):

    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))


    hor_line = cv2.line(frame, left_point, right_point, (255, 0, 0), 1)
    ver_line = cv2.line(frame, center_top, center_bottom, (255, 0, 0), 1)

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = (hor_line_length / ver_line_length)
    return ratio


Alaram=False
Total=0


cap=cv2.VideoCapture(0)

# initializes dlib’s pre-trained face detector so we can detect the face
detector=dlib.get_frontal_face_detector()

# loads the facial landmark predictor using the path to the supplied for detecting the landmarks points
predictor=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

font=cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # here we are detecting the face on the gray frame
    faces=detector(gray)

    for face in faces:

        # determines the facial landmarks for the face region
        landmarks = predictor(gray, face)

        for n in range(36,48):
            x=landmarks.part(n).x
            y=landmarks.part(n).y

            cv2.circle(frame,(x,y),1,(0,255,255),-1)


        left_eye=get_blinking_ratio([36,37,38,39,40,41],landmarks)
        right_eye=get_blinking_ratio([42,43,44,45,46,47,48],landmarks)
        blinking_ratio=((left_eye + right_eye)/2)



        if(blinking_ratio < 5.7):
            Total=0
            Alaram=False
            cv2.putText(frame,'EYES OPEN',(10,30),font,1,(255,0,0),1)
            stop_sound()

        else:
            Total +=1
            if Total > 20:
                if not Alaram:
                    Alaram=True
                    d=threading.Thread(target=start_sound)
                    d.setDaemon(True)
                    d.start()

            cv2.putText(frame,'EYES CLOSE',(10,30),font,1,(0,0,255),1)

    cv2.imshow('video',frame)
    key=cv2.waitKey(1)
    if key==27:
        break


cv2.destroyAllWindows()