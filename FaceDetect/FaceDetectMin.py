import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('Videos/2.mp4')
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.35)
while True:
    success, img1 = cap.read()
    img = cv2.resize(img1, (640, 360), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    cTime = time.time()
    if results.detections:
        for id, detection in enumerate(results.detections):
             # print(id,detection)
             # print(detection.score)
             # print(detection.location_data.relative_bounding_box)
             #mpDraw.draw_detection(img, detection)
             bboxC = detection.location_data.relative_bounding_box
             ih, iw, ic = img.shape
             bbox = int(bboxC.xmin*iw), int(bboxC.ymin*ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
             cv2.rectangle(img,bbox,(255,0,255),2)
             cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 3, (0, 225, 0), 2)
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,225,0), 2)
    cv2.imshow("frame", img)
    cv2.waitKey(1)
