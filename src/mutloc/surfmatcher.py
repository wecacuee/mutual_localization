'''
Feature-based image matching sample.

USAGE
  surfmatcher.py  [ <image1> <image2> ]

Extracted from opencv/samples/python2/find_obj.py
'''

import numpy as np
import cv2
import os
from mutloc.findmarkers import debug_img

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6

def correspondence_points(img1, img2, tag='c'):
    """
    correspondence_points(img1, img2) -> points2d1, points2d2, keypoint
    img1: Image (ndarray)
    img2: Another image (ndarray)
    points2d1: Nx2 array of 2d points in img1
    points2d2: Nx2 array of 2d points in img2
    keypoints: A zipped data structure of matching points
    """
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    detector = cv2.SURF(800)
    norm = cv2.NORM_L2
    flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)
    raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
    p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)

    if len(p1) >= 4:
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        print '%d / %d  inliers/matched' % (np.sum(status), len(status))
        status = status.reshape(-1) # flatten
        p1 = p1[status == 1]
        p2 = p2[status == 1]
        kp_pairs = [kp_pairs[i] for i in range(len(kp_pairs)) if status[i] == 1]
    else:
        # Just depend on the thresholding for filtering matches
        p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches, ratio=0.3)

    draw_correspondence_points(img1, img2, kp_pairs, tag=tag)
    return p1, p2, kp_pairs

def draw_correspondence_points(img1, img2, kp_pairs, tag='c'):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2) in zip(p1, p2):
        col = green
        cv2.circle(vis, (x1, y1), 2, col, -1)
        cv2.circle(vis, (x2, y2), 2, col, -1)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2) in zip(p1, p2):
        cv2.line(vis, (x1, y1), (x2, y2), green)

    cv2.imshow(tag, vis)
    debug_img(tag, vis)
    
def filter_matches(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs
   
if __name__ == '__main__':
    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
    opts = dict(opts)
    feature_name = opts.get('--feature', 'sift')
    fn1, fn2 = args[:2]
    img1 = cv2.imread(fn1, flags=0) 
    if img1 is None:
        raise ValueError("Unable to open {0}".format(fn1))
    img2 = cv2.imread(fn2, flags=0)
    if img2 is None:
        raise ValueError("Unable to open {0}".format(fn2))
    correspondence_points(img1, img2)
    cv2.waitKey()
    cv2.destroyAllWindows() 			
