import pyscreenshot as ImageGrab
import numpy as np
import cv2
from pydarknet import Detector, Image
import time
import pyautogui

net = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"), bytes("weights/yolov3.weights",
                                                                encoding="utf-8"), 0, bytes("cfg/coco.data", encoding="utf-8"))

face_cascade = cv2.CascadeClassifier(
    '/home/sachin/opencv/opencv-3.4.0/data/haarcascades/haarcascade_frontalface_default.xml')


def screen_record():
    while(True):
        printscreen = np.array(ImageGrab.grab(bbox=(1450, 50, 2100, 550)))

        for (x, y, w, h) in detect_face(cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY)):
            cv2.rectangle(printscreen, (x, y), (x+w, y+h), (255, 0, 0), 2)
            shoot_face((x+w)/2, (y+h)/2)

        # for cat, score, bounds in net.detect(Image(printscreen)):
        #     x, y, w, h = bounds
        #     cv2.rectangle(printscreen, (int(x - w / 2), int(y - h / 2)),
        #                   (int(x + w / 2), int(y + h / 2)), (255, 0, 0), thickness=2)

        cv2.imshow('window', cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def shoot_face(x, y):
    pyautogui.moveTo(x, y)
    # pyautogui.moveRel(None, -20)
    pyautogui.click()
    pyautogui.doubleClick()


def detect_face(gray):
    return(face_cascade.detectMultiScale(gray, 1.3, 5))


screen_record()
