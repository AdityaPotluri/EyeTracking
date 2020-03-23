import cv2
from eye import Eye
import dlib


cap=cv2.VideoCapture(0)
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("face_landmarks.dat")

while True:
    _,frame=cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=detector(gray)
    
    for face in faces:
        landmarks=predictor(gray,face)
        
        leftEye=Eye(1,frame,landmarks)
        rightEye=Eye(2,frame,landmarks)
        
        leftEye.gazeMarks()
        rightEye.gazeMarks()
        
    cv2.imshow("Frame",frame)

    key=cv2.waitKey(100)

    if key==ord('e'):
        break


cap.release()
cv2.destroyAllWindows()
