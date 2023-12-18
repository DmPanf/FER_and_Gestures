import cv2
import time
import os
from fer import FER
import HandTrackingModule as htm

wCam, hCam = 640, 480
#wCam, hCam = 1280, 720

cap = cv2.VideoCapture(-1)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "fingers" # name of the folder, where there are images of fingers
fingerList = os.listdir(folderPath) # list of image titles in 'fingers' folder
overlayList = []
for imgPath in fingerList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    overlayList.append(image)

pTime = 0

detector = htm.handDetector(detectionCon=0.75)
totalFingers = 0

emotion_detector = FER()

while True:
    sucess, img = cap.read()
    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    if lmList:
        fingersUp = detector.fingersUp()
        totalFingers = fingersUp.count(1)

    h, w, c = overlayList[totalFingers].shape
    img[0:h, 0:w] = overlayList[totalFingers]

    cTime = time.time()
    fps = 1/ (cTime-pTime)
    pTime = cTime

    emotions = emotion_detector.detect_emotions(img)
    for emotion in emotions:
        (x, y, w, h) = emotion["box"]
        emotions_dict = emotion["emotions"]
        dominant_emotion = max(emotions_dict, key=emotions_dict.get)
        emotion_score = emotions_dict[dominant_emotion]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        emotion_text = "{}: {:.2f}%".format(dominant_emotion, emotion_score * 100)
        cv2.putText(img, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:  # 'Esc'
        cap.release()
        cv2.destroyAllWindows()
        break

