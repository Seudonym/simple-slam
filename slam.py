#!/usr/bin/env python3
import cv2 as cv
import numpy as np

W = 1920 // 2
H = 1080 // 2

class FeatureExtractor:
    def __init__(self):
        self.orb = cv.ORB.create(200)
        self.bf = cv.BFMatcher(cv.NORM_HAMMING) # NORM_HAMMING should be used for ORB, check https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
        self.previous_extraction = None
    
    def extract_features(self, img):
        ### Detection
        # Get good corners to track from GFTT
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        corners = cv.goodFeaturesToTrack(gray, maxCorners=3000, qualityLevel=0.01, minDistance=3)

        ### Extraction
        # Use these corners as keypoints for ORB
        keypoints = [cv.KeyPoint(x=corner[0][0], y=corner[0][1], size=20) for corner in corners]
        keypoints, descriptors = self.orb.compute(img, keypoints)

        ### Matching
        # Use BFMatcher to match the descriptors with the previous frame
        if self.previous_extraction is None:
            self.previous_extraction = {'keypoints': keypoints, 'descriptors': descriptors}
            return keypoints, descriptors, None

        matches = self.bf.match(descriptors, self.previous_extraction['descriptors'])
        return keypoints, descriptors, matches

feature_extractor = FeatureExtractor()

def process_frame(img):
    img = cv.resize(img, (W, H))
    kps, des, matches = feature_extractor.extract_features(img)
    img2 = img.copy()
    cv.drawKeypoints(img, kps, img2, color=(0, 255, 0), flags=0)
    cv.imshow("frame", img2)
    if cv.waitKey(1) & 0xFF == ord('q'):
        return

if __name__ == "__main__":
    cap = cv.VideoCapture("./road1.mp4")
    while True:
        ret, frame = cap.read()
        if not ret: break
        process_frame(frame)

    cap.release()
