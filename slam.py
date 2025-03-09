#!/usr/bin/env python3
import cv2 as cv
import numpy as np
from utils import FeatureTracker
from random import randint

W = 1920 // 2
H = 1080 // 2

ft = FeatureTracker()
last_frame = None

def rand_color():
    return (randint(0, 255), randint(0, 255), randint(0, 255))

def process_frame(img):
    global last_frame
    img = cv.resize(img, (W, H))
    matches = ft.extract_features(img)
    if matches is None: return
    if last_frame is None:
        last_frame = img
        return


    for pt1, pt2 in matches:
        u1, v1 = map(lambda x: int(round(x)), pt1)
        u2, v2 = map(lambda x: int(round(x)), pt2)
        color = rand_color()
        cv.circle(img, center=(u1, v1), radius=2, color=color)
        cv.circle(img, center=(u2, v2), radius=2, color=color)
        cv.line(img, (u1, v1), (u2, v2), color=color, thickness=1)

    last_frame = img
    cv.imshow("matches", img)
    # if cv.waitKey(0):
    if cv.waitKey(1) & 0xFF == ord('q'):
        return

if __name__ == "__main__":
    cap = cv.VideoCapture("./road1.mp4")
    while True:
        ret, frame = cap.read()
        if not ret: break
        process_frame(frame)

    cap.release()
