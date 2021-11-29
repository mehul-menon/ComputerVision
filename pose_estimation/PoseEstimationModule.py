import cv2
import mediapipe as mp
import time
class PoseDetector():
    def __init__(self, mode=False, complexity=1, smooth_l=True, en_segm=False, smooth_segm=True, detectCon=0.5, trackCon=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth_l = True
        self.en_segm = en_segm
        self.smooth_segm = smooth_segm
        self.detectCon = detectCon
        self.trackCon = trackCon
        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(self.mode,self.complexity,self.smooth_l,self.en_segm,
                                     self.smooth_segm,self.detectCon,self.trackCon)


    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

    def findPosition(self,img,draw=True):
        lmlist = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                 h, w, c = img.shape
                 cx, cy = int(lm.x * w), int(lm.y * h)
                 lmlist.append([id,cx,cy])
                 cv2.circle(img, (cx, cy), 5, (0, 255, 255), cv2.FILLED)
        return lmlist
def main():
    pTime = 0
    cap = cv2.VideoCapture('PoseVideos/2.mp4')
    detector = PoseDetector()
    while True:
        success, img1 = cap.read()
        img = cv2.resize(img1, (640, 360), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        detector.findPose(img)
        detector.findPosition(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow("frame", img)
        cv2.waitKey(1)
if __name__=="__main__":
    main()