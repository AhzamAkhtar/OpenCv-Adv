import cv2 as cv
import mediapipe as mp
import time
#cap=cv.VideoCapture("video122.mp4")
cap=cv.VideoCapture(0)
pTime=0
mpDraw=mp.solutions.drawing_utils
mpFaceMesh=mp.solutions.face_mesh
#faceMesh=mpFaceMesh.FaceMesh()
faceMesh=mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec=mpDraw.DrawingSpec(thickness=1,circle_radius=1)
while True:
    sucess,img=cap.read()
    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results=faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,faceLms,mpFaceMesh.FACEMESH_CONTOURS,drawSpec,drawSpec)

        for id,lm in enumerate(faceLms.landmark):
            ih,iw,ic=img.shape
            x,y=int(lm.x*iw),int(lm.y*ih)
            cv.circle(img,(x,y),1,(0,255,0),thickness=2)
            print(id,x,y)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    img=cv.resize(img,(800,800))
    cv.putText(img,f"FPS : {int(fps)}",(20,70),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),thickness=4)
    cv.imshow("Images",img)
    if cv.waitKey(10)==ord("q"):
        break
cv.destroyWindow()