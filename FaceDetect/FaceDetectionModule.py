import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, detectCon=0.5, model_sel=0): #defining init file
        self.detectCon = detectCon # detection confidence
        self.model_sel = model_sel 
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(0.35)

    def findFace(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
             for id, detection in enumerate(self.results.detections):
             # print(id,detection)
             # print(detection.score)
             # print(detection.location_data.relative_bounding_box)
             #mpDraw.draw_detection(img, detection)
                 bboxC = detection.location_data.relative_bounding_box
                 ih, iw, ic = img.shape
                 bbox = int(bboxC.xmin*iw), int(bboxC.ymin*ih), \
                     int(bboxC.width * iw), int(bboxC.height * ih) #bbox returns size of bounding box as ratio with respect to image
                 bboxs.append([id,bbox,detection.score])
             if draw:
                 cv2.rectangle(img,bbox,(255,0,255),2)
                 cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 3, (0, 225, 0), 2)
        return img, bboxs
def main():
    cap = cv2.VideoCapture('Videos/2.mp4')
    pTime = 0
    detector = FaceDetector()
    while True:
        success, img1 = cap.read()
        img = cv2.resize(img1, (640, 360), fx=0, fy=0, interpolation=cv2.INTER_CUBIC) 
        img, bboxs = detector.findFace(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)# fps
        pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 225, 0), 2)
        cv2.imshow("frame", img)
        cv2.waitKey(1)
if __name__=="__main__":
    main()
