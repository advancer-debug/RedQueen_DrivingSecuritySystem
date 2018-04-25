import cv2
import numpy
import matplotlib.pyplot as plot



def getPhoto(cap):
    ret, frame = cap.read()
    # cv2.imshow("capture", frame)
    res = cv2.resize(src=frame, dsize=(640, 480) , interpolation = cv2.INTER_CUBIC)
    #
    # print(res.shape, type(res))
    # cv2.imshow("res", res)
    return res

