import cv2
import numpy as np
import dlib




class Eye(object):
    
    leftEye=[36,37,38,39,40,41]
    rightEye=[42,43,44,45,46,47]

    """
    The variables leftEye and right Eye are simply a list of landmark points for each eye as can bee seen in the image 
    attached in the folder. The class takes three arguements side (1 for left and 2 for right),a cv2 frame, and landmarks
    which can be created through the dlib library. It also calculates some new variables :
    
    region:the points on the cv2 frame for all the points on the eye
    Eyframe:the cv2 frame based only on the points in region
    center:the centroid of the region
    origin:the origin of the new frame (minimum x and y values of the region)

    The analyze eye method calculates for the 4 variables above


    """

    def __init__(self,side,frame,landmarks):
        assert(side==1 or side==2),"Invalid side given 1 is for left 2 for right"
        assert(isinstance(frame,np.ndarray)),"frame is not a numpy.ndarray"
        assert(isinstance(landmarks,dlib.dlib.full_object_detection)),"landmarks is not a dlib.dlib.full_object_detection"

        self.side=side
        self.fullFrame=frame
        self.landmarks=landmarks

        self.Eyeframe=None
        self.region=None
        self.analyzeEye()
        
        #implement a Pupil class to track Pupil movement given an eyeframe,region,center,origin
    
    def gazeMarks(self):
        if self.side==1:
            top1=Eye.midpoint(self.point(37),self.point(38))
            bottom1=Eye.midpoint(self.point(41),self.point(40))
            right1=self.point(39)
            left1=self.point(36)

            cv2.line(self.fullFrame,right1,left1,(255,0,0),1)   
            cv2.line(self.fullFrame,top1,bottom1,(255,0,0))   
            
            cv2.putText(self.fullFrame,self.gaze_direction(),(75,100),cv2.FONT_HERSHEY_PLAIN,0.5,(0,0,0))
            
        else:
            top2=Eye.midpoint(self.point(43),self.point(44))
            bottom2=Eye.midpoint(self.point(47),self.point(46))
            left2=self.point(42)
            right2=self.point(45)

            cv2.line(self.fullFrame,top2,bottom2,(0,0,255),1)
            cv2.line(self.fullFrame,left2,right2,(0,0,255),1)
            cv2.putText(self.fullFrame,self.gaze_direction(),(25,100),cv2.FONT_HERSHEY_PLAIN,0.5,(0,0,0))
        cv2.polylines(self.fullFrame,[self.region],True,(255,255,255))
        
        
        

    def analyzeEye(self):
        if self.side==1:
            self.region=np.array([self.point(p) for p in Eye.leftEye],dtype=np.int32)
        else:
            self.region=np.array([self.point(p) for p in Eye.rightEye],dtype=np.int32)
        
        self.Eyeframe=self.eyeFramePoints()
        
        #self.Eyeframe=cv2.resize(self.Eyeframe,None,fx=5,fy=5)
        

        
    
 
    def eyeFramePoints(self):
        minX=np.min(self.region[:,0])
        maxX=np.max(self.region[:,0])
        minY=np.min(self.region[:,1])
        maxY=np.max(self.region[:,1])
        return self.fullFrame[minY:maxY,minX:maxX]
        

    

    def point(self,landmark_num):
        return (self.landmarks.part(landmark_num).x,self.landmarks.part(landmark_num).y)
    
    
    def is_blinking(self):
        
        distance=lambda p1,p2:((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5
        if(self.side==1):
            top1=Eye.midpoint(self.point(37),self.point(38))
            bottom1=Eye.midpoint(self.point(41),self.point(40))
            right1=self.point(39)
            left1=self.point(36)

            height=distance(top1,bottom1)
            width=distance(right1,left1)

            return (width/height>6.7)
        else:
            top2=Eye.midpoint(self.point(43),self.point(44))
            bottom2=Eye.midpoint(self.point(47),self.point(46))
            left2=self.point(42)
            right2=self.point(45)

            height=distance(top2,bottom2)
            width=distance(left2,right2)

            return (width/height>6.7)

    #returns the ratio of the white part of the eye to the colored part of each half of an eye 
    def getIrisScleraRatio(self):
        
        gray_eye=cv2.cvtColor(self.Eyeframe,cv2.COLOR_BGR2GRAY)
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
        height, width = threshold_eye.shape

        left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
        sclera_left = cv2.countNonZero(left_side_threshold)

        right_side_threshold = threshold_eye[0: height, int(width / 2): width]
        sclera_right = cv2.countNonZero(right_side_threshold)
        try:
            return sclera_right/sclera_left
        except ZeroDivisionError:
            print("zero divison")
            return 1.3
        

    def gaze_direction(self):
        if(self.getIrisScleraRatio()<=0.9):
            return "Right"
        elif(0.9<self.getIrisScleraRatio()<1.8):
            return "Center"
        return "Left"
    
    #this method expects a tuple not landmark numbers to be passed in which can easily be done with the point method below
    @staticmethod
    def midpoint(p1,p2):
        x_value=(p1[0]+p2[0])//2
        y_value=(p1[1]+p2[1])//2
        return (x_value,y_value)

    
    def __repr__(self):
        print(f"Region::{self.region} \n Center::{self.center} Origin::{self.origin}")