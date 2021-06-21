"""
Based on:
Rectify images, create disparity maps, filter disparity maps, and create pointclouds. 
Camera calibration must be carried out beforehand.
"""

# Python 2/3 compatibility
from __future__ import print_function

import cv2
import numpy as np
import matplotlib.pyplot as plt

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

# tuning parameters for cv2.StereoSGBM
min_disp = 16
num_disp = 112-min_disp # must be divisible by 16
block_size = 9
disp12_max_diff = -1
uniqueness_ratio = 17
speckle_window_size = 200
speckle_range = 2 


def rectify(calibration_file, left_img, right_img):
    # open calibration file
    with np.load(calibration_file, allow_pickle=True) as X:
        [ret, mtx1, dist1, mtx2, dist2, R, T, E, F] = X['arr_0']

    lFrame = cv2.imread(left_img)
    rFrame = cv2.imread(right_img)
    h,w = lFrame.shape[:2] # both frames should be of same shape
    frames = [lFrame, rFrame]

    # Params from camera calibration
    camMats = [mtx1, mtx2]
    distCoeffs = [dist1, dist2]

    camSources = [0,1]
    for src in camSources:
        distCoeffs[src][0][4] = 0.0 # use only the first 2 values in distCoeffs

    # The rectification process
    newCams = [0,0]
    roi = [0,0]
    for src in camSources:
        newCams[src], roi[src] = cv2.getOptimalNewCameraMatrix(cameraMatrix = camMats[src], 
                                                               distCoeffs = distCoeffs[src], 
                                                               imageSize = (w,h), 
                                                               alpha = 0)

    rectFrames = [0,0]
    for src in camSources:
            rectFrames[src] = cv2.undistort(frames[src], 
                                            camMats[src], 
                                            distCoeffs[src])

    # See the results
    view = np.hstack([frames[0], frames[1]])    
    rectView = np.hstack([rectFrames[0], rectFrames[1]])

    #cv2.imwrite(r"Y:\Yolo_v4\darknet\build\darknet\x64\LAKSIT_calibration\rect\rect_left.png", rectFrames[0])
    #cv2.imwrite(r"Y:\Yolo_v4\darknet\build\darknet\x64\LAKSIT_calibration\rect\rect_right.png", rectFrames[1])

    h, w, _ = view.shape
    #cv2.imshow('view', cv2.resize(view, (w//2, h//2)))

    h, w, _ = rectView.shape

    # 10 lines evenly placed to check rectification
    color = (0, 0, 255)
    thickness = 3
    i = lineHeight = h//10
    while i < h:
        rectView = cv2.line(rectView, (0,i), (w,i), color, thickness)
        i += lineHeight

    rectView = cv2.resize(rectView, (w//2, h//2))
    #cv2.imshow('rectView', rectView)

    # Wait indefinitely for any keypress
    #cv2.waitKey(0)

    # return the recitified frames
    return rectFrames[0],rectFrames[1]

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

def disparity(rect1, rect2):
    # blur and downsample the rectified images
    rect1 = cv2.pyrDown(rect1)
    rect2 = cv2.pyrDown(rect2)

    # tuning parameters - need to be tuned properly for this case
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp # must be divisible by 16
    block_size = 9 #16

    #stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=15)

    stereo_left = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = block_size,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = disp12_max_diff, 
        uniquenessRatio = uniqueness_ratio,
        speckleWindowSize = uniqueness_ratio,
        speckleRange = speckle_range,
        mode = cv2.StereoSGBM_MODE_HH
    )
    print('computing disparity...')

    stereo_right = cv2.ximgproc.createRightMatcher(stereo_left)

    disp1 = stereo_left.compute(rect1, rect2).astype(np.float32) / 16.0
    disp2 = stereo_right.compute(rect2, rect1).astype(np.float32) / 16.0

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo_left)
    filtered_disp = wls_filter.filter(disparity_map_left=disp1, left_view=rect1, disparity_map_right=disp2)

    cv2.imshow("raw disparity", (disp1-min_disp)/num_disp)
    cv2.imshow("filtered disparity", (filtered_disp-min_disp)/num_disp)
    #cv2.waitKey()


    #cv2.imshow('disparity1', (disp1-min_disp)/num_disp)
    #cv2.imshow('disparity2', (disp2-min_disp)/num_disp)
    #cv2.waitKey(0)

    return disp1, disp2, filtered_disp

def pointcloud(disp,rect,filename):
    h, w = rect1.shape[:2]
    f = 11.2 #0.8*w  

    #cv2.imshow("rect1", rect) 
    #cv2.imshow("DISP", (disp-min_disp)/num_disp)
    #cv2.waitKey()

    Q = np.float32([[1, 0, 0, -525.1669464111328],
                    [0, 1, 0, -516.3065414428711],
                    [0, 0, 0, 2026.464289992846],
                    [0, 0, 0.003669942135328761, 0.8054465728326005]])


    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(cv2.pyrDown(rect), cv2.COLOR_BGR2RGB)
    print("colors", colors.shape)
    print("points", points.shape)
    mask = disp > min_disp*2
    print("disp", disp)
    print("disp type", type(disp))
    print("disp min", np.nanmin(disp))
    print("mask", mask)
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = filename
    print("out_points", out_points)
    print("out_colors", out_colors)
    write_ply(out_fn, out_points, out_colors)
    print('%s saved' % out_fn)


if __name__ == "__main__":
    calibration_file = r"Y:\Yolo_v4\darknet\build\darknet\x64\LAKSIT_calibration\calibration\calibration_params.npz"
    left_img = r"Y:\FishDetectionAndTracking\random_frames_2\stereo_frame144_out1.jpg"
    right_img = r"Y:\FishDetectionAndTracking\random_frames_2\stereo_frame144_out2.jpg"

    num = right_img.split("stereo_frame")[1].split("_out2")[0]
    pointcloud_file = "pointclouds/out_" + num + "_new.ply"

    rect1, rect2 = rectify(calibration_file, left_img, right_img)
    disp1, disp2, filtered_disp = disparity(rect1, rect2)
    #pointcloud(disp1, rect1)
    pointcloud(filtered_disp, rect1, pointcloud_file)

    cv2.destroyAllWindows()
