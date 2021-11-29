import mediapipe as mp
import time
import cv2

cap = cv2.VideoCapture(0)#webcam number zero

mpHands = mp.solutions.hands
hands = mpHands.Hands()
#ctrl and click for parameters. static image mode at false means the tracking is done only with minimum confidence level
#this is faster than when it is true. brackets are empty to use default values

mpDraw = mp.solutions.drawing_utils

pTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB) # to check use print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:#checking if landmarks are detected
        for x in results.multi_hand_landmarks:#looping over landmarks
            for id, lm in enumerate(x.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h) #lm.x and lm.y store values as ratios with the total length or width. we multiply to get pixels
                print(id, cx, cy)
                if id==0:#showing specific landmark
                    cv2.circle(img, (cx, cy),15, (255,255,0), cv2.FILLED)
            mpDraw.draw_landmarks(img,x,mpHands.HAND_CONNECTIONS)#3rd parameter draws connecting lines

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 3,(255, 0, 0), 2)

    cv2.imshow("frame",img)
    cv2.waitKey(1)
