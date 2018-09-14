# coding=utf-8
import cv2
import numpy as np
import operator
import collections
import math
import sys
import matplotlib.pyplot as plt
import os
import segmentDoc

vid=cv2.VideoCapture("testVideo.mp4")
while True:
    ret,frame=vid.read()
    if ret:
        image=segmentDoc.draw(frame)
    # print ret
        cv2.imshow('Video',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
vid.release()
cv2.destroyAllWindows()