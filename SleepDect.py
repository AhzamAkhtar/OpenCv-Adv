import cv2 as cv
import mediapipe as mp
import time
import math
import winsound
#cap=cv.VideoCapture("video122.mp4")
cap=cv.VideoCapture(0)
pTime=0
mpDraw=mp.solutions.drawing_utils
mpFaceMesh=mp.solutions.face_mesh
#faceMesh=mpFaceMesh.FaceMesh()
faceMesh=mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec=mpDraw.DrawingSpec(thickness=1,circle_radius=1)
cx159,cy159=0,0
cx145,cy145=0,0
cx386,cy386=0,0
cx374,cy374=0,0
while True:
    sucess,img=cap.read()
    original_frame=img.copy()
    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results=faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            for id,lm in enumerate(faceLms.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                if id==159:
                    cx159,cy159=cx,cy
                    cv.circle(img, (cx, cy), 1, (0, 0, 255), thickness=1)
                if id==145:
                    cx145,cy145=cx,cy
                    cv.circle(img, (cx, cy), 1, (0, 0, 255), thickness=1)
                if id==386:
                    cx386,cy386=cx,cy
                    cv.circle(img, (cx, cy), 1, (0, 0, 255), thickness=1)
                if id==374:
                    cx374,cy374=cx,cy
                    cv.circle(img,(cx,cy),1,(0,0,255),thickness=1)
                #print(cx,cy)
                cv.line(img,(cx159,cy159),(cx145,cy145),(0,255,0),thickness=1)
                cv.line(img,(cx386,cy386),(cx374,cy374),(0,255,0),thickness=1)
                length_left=math.hypot(cx159-cx145,cy159-cy145)
                length_rigth=math.hypot(cx386-cx374,cy386-cy374)
                #print(int(length_left),int(length_rigth))
                if length_rigth and length_left<=10:
                    print("alert")
                    #winsound.Beep(50,50)
            #cv.circle(img, (cx, cy), 1, (0, 255, 0), thickness=1)
            #mpDraw.draw_landmarks(img,faceLms,mpFaceMesh.FACEMESH_CONTOURS,drawSpec,drawSpec)

        #for id,lm in enumerate(faceLms.landmark):
            #ih,iw,ic=img.shape
            #x,y=int(lm.x*iw),int(lm.y*ih)
            #cv.circle(img,(x,y),1,(0,255,0),thickness=1)
            #print(id,x,y)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    img=cv.resize(img,(800,800))
    cv.putText(img,f"FPS : {int(fps)}",(20,70),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),thickness=4)
    #cv.imshow("Images",original_frame)
    cv.imshow("Images",img)
    if cv.waitKey(10)==ord("q"):
        break
cv.destroyWindow()