import time
import cv2 as cv
import numpy as np
import mediapipe as mp
import math


class HandDetector:
    def __init__(self, mode=False, maxHand=2, detectCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHand = maxHand
        self.detectCon = detectCon
        self.trackCon = trackCon

        self.mpHand = mp.solutions.hands
        self.Hands = self.mpHand.Hands(self.mode, self.maxHand, 1, self.detectCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findhands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.processHand = self.Hands.process(imgRGB)
        if self.processHand.multi_hand_landmarks:
            for handLMS in self.processHand.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLMS, self.mpHand.HAND_CONNECTIONS)

    def findPosition(self, img, handsNo=0):
        self.lmlist = []
        xList = []
        ylist = []
        bbox = []
        if self.processHand.multi_hand_landmarks:
            try:
                myHand = self.processHand.multi_hand_landmarks[handsNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmlist.append([id, cx, cy])
                    xList.append(cx)
                    ylist.append(cy)
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(ylist), max(ylist)
                bbox = xmin, ymin, xmax, ymax
            except:
                pass

        return self.lmlist, bbox

    def fingersUp(self):
        fingers = []
        if self.lmlist[self.tipIds[0]][1] > self.lmlist[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if self.lmlist[self.tipIds[id]][2] < self.lmlist[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmlist[p1][1:]
        x2, y2 = self.lmlist[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv.circle(img, (x1, y1), r, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), r, (255, 0, 255), cv.FILLED)
            cv.circle(img, (cx, cy), r, (0, 0, 255), cv.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, [x1, y1, x2, y2, cx, cy]


def main():
    cap = cv.VideoCapture(0)
    detector = HandDetector()
    while True:
        sucess, img = cap.read()
        detector.findhands(img)
        lmLIST, bbox = detector.findPosition(img)

        if len(lmLIST) != 0:
            fingers = detector.fingersUp()
            length, bbox = detector.findDistance(8, 12, img)
            print(length)
            print(fingers)

            # New feature: Display the count of fingers detected
            total_fingers = fingers.count(1)
            cv.putText(img, f'Fingers: {total_fingers}', (10, 70), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        cv.imshow("result", img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break


# Code by Vigneshwaran
if __name__ == '__main__':
    main()
