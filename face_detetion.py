import cv2 as cv
import mediapipe as mp
import time
pTime=0
#cap=cv.VideoCapture("video122.mp4")
cap=cv.VideoCapture(0)
mpFaceDetection=mp.solutions.face_detection
faceDetection=mpFaceDetection.FaceDetection()
mpDraw=mp.solutions.drawing_utils
while True:
    sucess,img=cap.read()
    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results=faceDetection.process(imgRGB)
    if results.detections:
        for id,detection in enumerate(results.detections):
            #print(id,detection)
            print(detection.location_data.relative_bounding_box)
            #mpDraw.draw_detection(img,detection)
            bboxC=detection.location_data.relative_bounding_box
            ih,iw,ic=img.shape
            bbox=int(bboxC.xmin*iw),int(bboxC.ymin*ih),int(bboxC.width*iw),int(bboxC.height*ih)
            cv.rectangle(img,bbox,(255,0,255),4)
            cv.putText(img, f"{int(detection.score[0]*100)}%", (bbox[0],bbox[1]-20), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), thickness=5)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv.putText(img,f"FPS: {int(fps)}",(20,70),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),thickness=5)
    img=cv.resize(img,(1000,1000))
    cv.imshow("Images", img)
    if cv.waitKey(20)==ord("q"):
        break
cv.destroyWindow()