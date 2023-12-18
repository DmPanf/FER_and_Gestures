import cv2
import time
import os
from fer import FER
import HandTrackingModule as htm

#wCam, hCam = 640, 480
wCam, hCam = 1280, 720

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

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None
is_recording = False

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

    #""" ## Emotions
    emotions = emotion_detector.detect_emotions(img)
    for emotion in emotions:
        (x, y, w, h) = emotion["box"]
        emotions_dict = emotion["emotions"]
        dominant_emotion = max(emotions_dict, key=emotions_dict.get)
        emotion_score = emotions_dict[dominant_emotion]
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        x_new, y_new = int(x - w * 0.2+20), int(y - h * 0.4+20)
        w_new, h_new = int(w * 1.3), int(h * 1.7)
        if dominant_emotion == 'angry':
            color = (0, 0, 255)  # RED
        elif dominant_emotion == 'happy':
            color = (0, 255, 0)  # Light-Green
        else:
            color = (255, 0, 0)  # DarkBlue
        cv2.ellipse(img, (x_new + w_new // 2, y_new + h_new // 2), (w_new // 2, h_new // 2), 0, 0, 360, color, 20)
        emotion_text = "{}: {:.2f}%".format(dominant_emotion, emotion_score * 100)
        cv2.putText(img, emotion_text, (20, 550 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 6) # (210, 255, 210)
    #"""
        
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    cv2.imshow("DMITRII PANFILOV", img)
    key = cv2.waitKey(1) & 0xFF

    #if cv2.waitKey(1) & 0xFF == 27:  # 'Esc'
    #    cap.release()
    #    cv2.destroyAllWindows()
    #    break

    # Начать/остановить запись при нажатии пробела
    if key == 32:  # ASCII код для пробела
        if is_recording:
            is_recording = False
            out.release()  # Закрыть файл записи
        else:
            is_recording = True
            out = cv2.VideoWriter('output.avi', fourcc, 20.0, (wCam, hCam))

    # Запись кадра, если запись активна
    if is_recording:
        out.write(img)

    if key == 27:  # 'Esc'
        break

# Освобождение ресурсов
cap.release()
if is_recording:
    out.release()
cv2.destroyAllWindows()
