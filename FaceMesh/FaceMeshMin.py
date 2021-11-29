import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('Videos/4.mp4')
pTime = 0
#428 points
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1,circle_radius=2)
while True:
    success, img1 = cap.read()
    img = cv2.resize(img1, (640, 360), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec)
            for id, lm in enumerate(faceLms.landmark):
                ih, iw, ic = img.shape
                cx, cy = int(lm.x*iw), int(lm.y*ih)
                print(id, cx, cy)
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
    cv2.imshow("frame",img)
    cv2.waitKey(1)