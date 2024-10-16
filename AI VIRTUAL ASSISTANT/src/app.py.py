import HandTrackingModule as htm
import cv2 as cv
import numpy as np
import autopy

# Author: Vigneshwaran

wCam, hCam = 640, 480
wScr, hScr = autopy.screen.size()
frameR = 150
smootheing = 5
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.HandDetector()

def Mouse(img):
    global frameR, smootheing, plocX, plocY, clocX, clocY, wScr, wCam, hScr, hCam
    detector.findhands(img)
    lmlist, bbox = detector.findPosition(img)

    cv.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

    if len(lmlist) != 0:
        Xindex, Yindex = lmlist[8][1], lmlist[8][2]
        Xmidel, Ymidel = lmlist[12][1], lmlist[12][2]
        fingers = detector.fingersUp()

        if fingers[1] == 1 and fingers[2] == 0:
            xMOUSE = np.interp(Xindex, (frameR, wCam - frameR), (0, wScr))
            yMOUSE = np.interp(Yindex, (frameR, hCam - frameR), (0, hScr))
            clocX = plocX + (xMOUSE - plocX) / smootheing
            clocY = plocY + (yMOUSE - plocY) / smootheing
            autopy.mouse.move(clocX, clocY)
            cv.circle(img, (Xindex, Yindex), 15, (20, 180, 90), cv.FILLED)
            plocY, plocX = clocY, clocX

        if fingers[1] == 1 and fingers[2] == 1:
            length, bbox = detector.findDistance(8, 12, img)
            if length < 40:
                autopy.mouse.click()

        
        cv.putText(img, f'Fingers Up: {sum(fingers)}', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return img

def main():
    while True:
        sucess, img = cap.read()
        img = cv.flip(img, 1)

        img = Mouse(img)

        cv.imshow("result", img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()
