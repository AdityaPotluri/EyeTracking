import cv2
import numpy as np
import dlib
from eye import Eye



cap=cv2.VideoCapture(0)
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("face_landmarks.dat")

while True:
    _,frame=cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=detector(gray)
    
    for face in faces:
        landmarks=predictor(gray,face)
        
        a=Eye(1,frame,landmarks)
        b=Eye(1,frame,landmarks)
        
        a.showEye()
        
    cv2.imshow("Frame",frame)

    key=cv2.waitKey(1)

    if key==ord('e'):
        break


cap.release()
cv2.destroyAllWindows()
