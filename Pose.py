import cv2 as cv
import time
import mediapipe as mp
pTime=0
cap=cv.VideoCapture("production ID_3943396.mp4")
#cap=cv.VideoCapture(0)
mpDraw=mp.solutions.drawing_utils
mpPose=mp.solutions.pose
pose=mpPose.Pose()
while True:
    sucess,img=cap.read()
    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results=pose.process(imgRGB)
    #print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id,lm in enumerate(results.pose_landmarks.landmark):
            h,w,c=img.shape
            print(id,lm)
            cx,cy=int(lm.x*w),int(lm.y*h)
            cv.circle(img,(cx,cy),5,(255,0,0),thickness=-1)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv.putText(img,str(int(fps)),(70,50),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv.imshow("Image", img)
    if cv.waitKey(10)==ord("q"):
        break
cv.destroyWindow()
