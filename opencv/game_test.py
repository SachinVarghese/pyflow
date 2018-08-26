import pyscreenshot as ImageGrab
import numpy as np
import cv2
from pydarknet import Detector, Image
import time

net = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"), bytes("weights/yolov3.weights",
                                                                encoding="utf-8"), 0, bytes("cfg/coco.data", encoding="utf-8"))


def detectFrame(img):
    return(net.detect(img))


def test():
    # printscreen = np.array(ImageGrab.grab(bbox=(0, 40, 200, 300)))
    printscreen = cv2.imread('cs.jpg')
    last_time = time.time()
    results = detectFrame(Image(printscreen))
    print('loop took {} seconds'.format(time.time()-last_time))
    last_time = time.time()
    for cat, score, bounds in results:
        x, y, w, h = bounds
        cv2.rectangle(printscreen, (int(x - w / 2), int(y - h / 2)),
                      (int(x + w / 2), int(y + h / 2)), (255, 0, 0), thickness=2)
        # cv2.putText(img, str(cat.decode("utf-8")), (int(x), int(y)),cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))

    cv2.imshow('window', cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)


test()
