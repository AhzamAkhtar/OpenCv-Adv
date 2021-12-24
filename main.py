#pip install opencv-contrib-python
#pip install mediapipe

# ___ Hand Tracking _____
import time

import cv2 as cv
import mediapipe as mp
wCam,hCam=640,480
cap=cv.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils

# To get FPS
ptime=0
ctime=0


while True:
    sucess,img=cap.read()
    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                #print(id,lm)
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                print(id,cx,cy)
                #if id==4:
                cv.circle(img,(cx,cy),25,(255,0,0),-1)
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)
            # the first value in () is where do we  want to put the results
            # and second for when we have multiple hands
            # and the third one is to draw the connection or lines b/t the 21 landmarks
    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv.imshow("Images",img)
    if cv.waitKey(10)==ord("q"):
        break
cv.destroyWindow()