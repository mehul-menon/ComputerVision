import cv2
import mediapipe as mp
import time
#428 points
class FaceMeshDetector():
    def __init__(self, mode=False, max_face=2, detectCon=0.5, trackCon=0.5):
        self.mode = mode
        self.max_face = max_face
        self.detectCon = detectCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=2)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1,circle_radius=2)

    def MeshDetector(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        faces = []
        if results.multi_face_landmarks:

            for faceLms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec)
                    face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    cx, cy = int(lm.x*iw), int(lm.y*ih)
                    face.append([id, cx, cy])
            faces.append((face))
        return img, faces

def main():
    #cap = cv2.VideoCapture('Videos/4.mp4')
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img1 = cap.read()
        img = cv2.resize(img1, (640, 360), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        img, face = detector.MeshDetector(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow("frame", img)
        cv2.waitKey(1)
if __name__=="__main__":
    main()