# Python 2/3 compatibility
from __future__ import print_function

import cv2
import numpy as np
#import matplotlib.pyplot as plt

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

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def rectify(calibration_file, left_img, right_img):
    """
    Takes left and right image paths, outputs left and right recitified images.
    """
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
    rectFrames = [0,0]
    for src in camSources:
            rectFrames[src] = cv2.undistort(src=frames[src], 
                                            cameraMatrix=camMats[src], 
                                            distCoeffs=distCoeffs[src])
    
    # See the results
    view = np.hstack([frames[0], frames[1]])    
    rectView = np.hstack([rectFrames[0], rectFrames[1]])

    h, w, _ = rectView.shape

    # 42 lines evenly placed to check rectification
    color = (0, 255, 0)
    thickness = 1
    i = lineHeight = h//42
    while i < h:
        rectView = cv2.line(rectView, (0,i), (w,i), color, thickness)
        i += lineHeight

    rectView = cv2.resize(rectView, (w//2, h//2))
    cv2.imshow('rectView', rectView)
    cv2.waitKey()

    # return the recitified frames
    return rectFrames[0],rectFrames[1]

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

def disparity(rect1, rect2, filter=False):
    # blur and downsample the rectified images
    rect1 = cv2.pyrDown(rect1)
    rect2 = cv2.pyrDown(rect2)

    # tuning parameters - need to be tuned properly for this case
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp # must be divisible by 16
    block_size = 12

    stereo_left = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = block_size,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = -1,
        uniquenessRatio = 5,
        speckleWindowSize = 10,
        speckleRange = 32,
        mode = cv2.StereoSGBM_MODE_HH
    )

    stereo_right = cv2.ximgproc.createRightMatcher(stereo_left)

    disp1 = stereo_left.compute(rect1, rect2).astype(np.float32) / 16.0
    disp2 = stereo_right.compute(rect2, rect1).astype(np.float32) / 16.0

    if filter:
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo_left)
        filtered_disp = wls_filter.filter(disparity_map_left=disp1, left_view=rect1, disparity_map_right=disp2)
    else:
        filtered_disp = disp1

    return disp1, disp2, filtered_disp, num_disp, min_disp

def pointcloud(disp,min_disp,rect,filename,Q):
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(cv2.pyrDown(rect), cv2.COLOR_BGR2RGB)
    mask = disp > min_disp
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = filename
    write_ply(out_fn, out_points, out_colors)
    print('%s saved' % out_fn)