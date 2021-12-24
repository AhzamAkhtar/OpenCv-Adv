#pip install opencv-contrib-python
#pip install mediapipe

# ___ Hand Tracking _____
import time
import os
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
cx2,cy2=0,0
cx3,cy3=0,0
cx6,cy6=0,0
cx8,cy8=0,0
cx12,cy12=0,0
cx10,cy10=0,0
cx16,cy16=0,0
cx14,cy14=0,0
cx20,cy20=0,0
cx18,cy18=0,0
# IMages
folderPath="FingerImages"
myList=os.listdir(folderPath)
#print(myList)
overlayList=[]
fingers=[]
for imPath in myList:
    images=cv.imread(f"{folderPath}/{imPath}")
    #print(f"{folderPath}/{imPath}")
    overlayList.append(images)
#print(len(overlayList))
tipIds=[4,8,12,16,20]
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
                #print(id,cx,cy)
                if id==3:
                   cx3,cy3=cx,cy
                if id==2:
                    cx2,cy2=cx,cy
                    if cx3>cx2:
                        output=1
                    else:
                        output=0
                if id==6:
                    cx6,cy6=cx,cy
                if id==8:
                    cx8, cy8 = cx, cy
                    if cy8<cy6:
                        output1=1
                    else:
                        output1=0
                if id==10:
                    cx10,cy10=cx,cy
                if id==12:
                    cx12,cy12=cx,cy
                    if cy12<cy10:
                        output2 = 1
                    else:
                        output2 = 0
                if id==14:
                    cx14,cy14=cx,cy
                if id==16:
                    cx16,cy16=cx,cy
                    if cy16<cy14:
                        output3 = 1
                    else:
                        output3 = 0
                if id==18:
                    cx18,cy18=cx,cy
                if id==20:
                    cx20,cy20=cx,cy
                    if cy20<cy18:
                        output4 = 1
                    else:
                        output4 = 0
        if output==1:
            fingers.insert(0,1)
        else:
            fingers.insert(0,0)
        if output1==1:
            fingers.insert(1,1)
        else:
            fingers.insert(1,0)
        if output2==1:
            fingers.insert(2,1)
        else:
            fingers.insert(2,0)
        if output3 ==1:
            fingers.insert(3,1)
        else:
            fingers.insert(3,0)
        if output4==1:
            fingers.insert(4,1)
        else:
            fingers.insert(4,0)
        #print(fingers[:5])
        final=fingers[:5]
        totalFingers=final.count(1)
        print(totalFingers)
        #h, w, c = overlayList[0].shape
        #img[0:h, 0:w] = overlayList[totalFingers]
        mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)
            # the first value in () is where do we  want to put the results
            # and second for when we have multiple hands
            # and the third one is to draw the connection or lines b/t the 21 landmarks
    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv.putText(img,str(int(fps)),(400,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv.imshow("Images",img)
    #print(fingers)
    if cv.waitKey(10)==ord("q"):
        break
cv.destroyWindow()