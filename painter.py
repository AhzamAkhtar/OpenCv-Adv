import cv2 as cv
import mediapipe as mp
import os
import time
import numpy as np
folderpath="header"
mylist=os.listdir(folderpath)
#print(mylist)
overlay=[]
fingers=[]
for imPath in mylist:
    image=cv.imread(f"{folderpath}/{imPath}")
    overlay.append(image)
#print(len(overlay))
##-----------------------------------------------##
cap=cv.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils
header=overlay[0]
Cx8,Cy8=0,0
Cx12,Cy12=0,0
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
Tx,Ty=0,0
# To get FPS
ptime=0
ctime=0
###_________________##########
brushthickness=5
eraserthickness=50
drawcolor=(255,0,255)
xp,yp=0,0
imgcanvas=np.zeros((720,1280,3),np.uint8)
while True:
    # 1. Import Image
    # 2. Find HAnd Landmarks
    # 3. Check which finger is up
    # 4. If Selection Mode -- Two fingers are up
    # 5. If Drawing Mode -- Index fingers is up
    sucess,img=cap.read()
    img=cv.flip(img,1)
    img[0:125,0:1280]=header
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
                if id==8:
                    Cx8,Cy8=cx,cy
                    Tx, Ty = cx,cy
                    #cv.circle(img,(cx,cy),25,(255,0,0),-1)
                #mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)

                if id==12:
                    Cx12,Cy12=cx,cy
                    #cv.circle(img,(cx,cy),25,(255,0,0),-1)
                #mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)

                ###---------------------#############
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
        #print(final)
        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        ##_______________________#
        if final[1]==1 and final[2]==1:
            xp,yp=0,0
            print("selection Mode")
            if cy<125:
                if 250<cx<750:
                    header=overlay[0]
                    drawcolor=(255,0,255)
                if 550<cx<950:
                    header=overlay[1]
                    drawcolor=(255,0,0)
                if 800<cx<950:
                    header=overlay[2]
                    drawcolor=(0,255,0)
                if 1050<cx<1200:
                    header=overlay[3]
                    drawcolor=(0,0,0)
            cv.rectangle(img, (Cx8, Cy8 - 25), (Cx12, Cy12 + 25), drawcolor, thickness=-1)
        if final[1]==1 and final[2]!=1:
            #cv.circle(img,(Cx8,Cy8),15,drawcolor,thickness=-1)
            print("Drawing Mode")
            if xp==0 and yp==0:
                xp,yp=cx,cy
            if drawcolor==(0,0,0):
                cv.line(img, (xp, yp), (Tx, Ty), drawcolor, thickness=eraserthickness)
                cv.line(imgcanvas, (xp, yp), (Tx, Ty), drawcolor, thickness=eraserthickness)
            cv.line(img,(xp,yp),(Tx,Ty),drawcolor,thickness=brushthickness)
            cv.line(imgcanvas,(xp,yp),(Tx,Ty),drawcolor,thickness=brushthickness)
            xp,yp=Tx,Ty
    imgGray=cv.cvtColor(imgcanvas,cv.COLOR_BGR2GRAY)
    _,imgInv=cv.threshold(imgGray,50,255,cv.THRESH_BINARY_INV)
    imgInv=cv.cvtColor(imgInv,cv.COLOR_GRAY2RGB)
    img=cv.bitwise_and(img,imgInv)
    img=cv.bitwise_or(img,imgcanvas)
    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv.imshow("Images",img)
    cv.imshow("Images_2",imgcanvas)
    cv.imshow("INv",imgInv)
    if cv.waitKey(10)==ord("q"):
        break
cv.destroyWindow()
