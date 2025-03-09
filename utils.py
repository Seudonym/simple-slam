import cv2 as cv
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

class FeatureTracker:
    def __init__(self):
        self.orb = cv.ORB.create(200)
        self.bf = cv.BFMatcher(cv.NORM_HAMMING) # NORM_HAMMING should be used for ORB, check https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
        self.previous_extraction = None
    
    def extract_features(self, img):
        ### Detection
        # Get good corners to track from GFTT
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        corners = cv.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.01, minDistance=5)

        ### Extraction
        # Use these corners as keypoints for ORB
        keypoints = [cv.KeyPoint(x=corner[0][0], y=corner[0][1], size=20) for corner in corners]
        keypoints, descriptors = self.orb.compute(img, keypoints)

        ### Matching
        # Use BFMatcher to match the descriptors with the previous frame
        if self.previous_extraction is None:
            self.previous_extraction = {
                'keypoints': keypoints, 
                'descriptors': descriptors
            }
            return None
        matches = self.bf.knnMatch(descriptors, self.previous_extraction['descriptors'], k=2)
        goodKeypoints = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                kp1 = keypoints[m.queryIdx].pt
                kp2 = self.previous_extraction['keypoints'][m.trainIdx].pt
                goodKeypoints.append((kp1, kp2))

        
        ### RANSAC to eliminate outliers
        if len(goodKeypoints) > 0:
            goodKeypoints = np.array(goodKeypoints)
            model, inliers = ransac(
                    (goodKeypoints[:, 0], goodKeypoints[:, 1]), 
                    FundamentalMatrixTransform, 
                    min_samples=8, 
                    residual_threshold=1, 
                    max_trials=100
            )
            goodKeypoints = goodKeypoints[inliers]

        return goodKeypoints

