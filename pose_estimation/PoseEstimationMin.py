import cv2
import mediapipe as mp
import time
pTime = 0
mpPose = mp.solutions.pose
pose = mpPose.Pose()
cap = cv2.VideoCapture('PoseVideos/2.mp4')
mpDraw = mp.solutions.drawing_utils
while True:
    success, img1 = cap.read()
    img = cv2.resize(img1, (640,360),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img,(cx,cy),5,(0,255,255),cv2.FILLED)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
    cv2.imshow("frame", img)
    cv2.waitKey(1)
