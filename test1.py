#pip install opencv-contrib-python
#pip install mediapipe

# ___ Volume Tracking _____
import time
import math
import cv2 as cv
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
wCam,hCam=640,480
cap=cv.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils
lml4id=[]
lml4cx=[]
lml4cy=[]
# To get FPS
ptime=0
ctime=0
cx1=0
cy1=0
cx2=0
cy2=0
vol=0
volBar=400
volPer=0
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volRange=volume.GetVolumeRange()
minVol=volRange[0]
maxVol=volRange[1]
while True:
    sucess,img=cap.read()
    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    #print(id, cx1, cy1)
                    if id==4:
                        cx1,cy1=cx,cy
                        cv.circle(img,(cx1,cy1),15,(255,0,255),thickness=-1)
                    if id==8:
                        cx2,cy2=cx,cy
                        cv.circle(img, (cx2, cy2), 15, (255, 0, 255), thickness=-1)
                    #cv.circle(img,(cx,cy),20,(255,0,0),thickness=-1)
                    #cv.line(img,(0,0),(cx,cy),(0,0,255),thickness=2)
            cv.line(img,(cx1,cy1),(cx2,cy2),(0,0,255),thickness=3)
            Cx,Cy=(cx1+cx2)//2 , (cy1+cy2)//2
            cv.circle(img,(Cx,Cy),15,(255,0,255),thickness=-1)
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)
            length=math.hypot(cx2-cx1,cy2-cy1)
            #print(length)
            #  Hand Range 50 ot 300
            # Volume Range -65 to 0
            vol=np.interp(length,[50,300],[minVol,maxVol])
            volBar=np.interp(length,[50,300],[400,150])
            volPer=np.interp(length,[50,300],[0,100])
            print(int(length),vol)
            volume.SetMasterVolumeLevel(vol, None)

            if length<=50:
                cv.circle(img, (Cx, Cy), 15, (0, 255, 0), thickness=-1)
            # the first value in () is where do we  want to put the results
            # and second for when we have multiple hands
            # and the third one is to draw the connection or lines b/t the 21 landmarks
    cv.rectangle(img,(50,150),(85,400),(0,255,0),thickness=3)
    cv.rectangle(img,(50,int(volBar)),(85,400),(0,255,0),thickness=-1)
    cv.putText(img,f"{int(volPer)}%", (40, 450), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv.imshow("Images",img)
    if cv.waitKey(10)==ord("q"):
        break
cv.destroyWindow()