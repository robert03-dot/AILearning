import time
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
# Computer Vision solutions enable computers to recognise objects in digital images and videos.
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmark)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                if id == 0:
                    cv2.circle(img,
                               (cx, cy),
                               25,
                               (0, 0, 255),
                               cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)),
                (10, 70),
                cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 255),
                3)
    # waitkey method allows you to display a window that has been shown with the cv2.
    # imshow method for a specified period of time
    cv2.imshow('Image', img)
    cv2.waitKey(1)
