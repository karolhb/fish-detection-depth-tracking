"""
Based on: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
Calibrate the two cameras individually and save parameters to .npz file.

TBD: Make this into functions or classes to be able to calibrate N cameras?
"""

import numpy as np
from numpy import savez_compressed
import cv2
import glob
import pandas as pd
from configparser import ConfigParser
from matplotlib import pyplot as plt

# checkboard dimensions
x,y = 7,4 

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all the images.
objpointsL = [] # 3d point in real world space
objpointsR = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((x*y,3), np.float32)
objp[:,:2] = np.mgrid[0:x,0:y].T.reshape(-1,2)

## LEFT CAMERA
imagesL = glob.glob('Y:/Yolo_v4/darknet/build/darknet/x64/LAKSIT_calibration/caliimages/*cam00.png', recursive=False)


for fname in imagesL:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)
    #cv2.waitKey(0)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (x,y),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpointsL.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpointsL.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (x,y), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((x*y,3), np.float32)
objp[:,:2] = np.mgrid[0:x,0:y].T.reshape(-1,2)


## RIGHT CAMERA
imagesR = glob.glob('Y:/Yolo_v4/darknet/build/darknet/x64/LAKSIT_calibration/caliimages/*cam01.png', recursive=False)

for fname in imagesR:
    img = cv2.imread(fname)
    h,w = img.shape[:2]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)
    #cv2.waitKey(0)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (x,y),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpointsR.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpointsR.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (x,y), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

print(len(objpointsL) == len(objpointsR))

## Calibration
dataL = cv2.calibrateCamera(objpointsL, imgpointsL, (w,h),None,None)
_, mtxL, distL, _, _ = dataL
dataR = cv2.calibrateCamera(objpointsR, imgpointsR, (w,h),None,None)
_, mtxR, distR, _, _ = dataR

data = cv2.stereoCalibrate(objpointsL, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, (w,h),None,None)

retval, mtx1, dist1, mtx2, dist2, R, T, E, F = data

savez_compressed('LAKSIT_calibration/calibration/calibration_params.npz', data)