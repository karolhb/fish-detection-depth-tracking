# -------------------------------------------------------------------
# Main program for YOLO detection and IOU tracking
# Written by Karoline H. Barstein
# Partly inspired by Augmented Startups:
# (https://github.com/augmentedstartups/YOLOv4-Tutorials)
# -------------------------------------------------------------------

from ctypes import *
import os
import cv2
import shelve
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from tracker_utils import convert_back, convert_tracks
from iou_trackers import tracker_iou, tracker_iou_allow_gaps
from rectification_and_disparity import rectify, disparity

import sys
sys.path.append(os.path.join(os.getcwd(),'python/'))
import darknet

def extract_coords(detections):
    centroids = dict()
    if len(detections) > 0:
        objectId = 0
        for detection in detections:           
            x, y, w, h, score = detection[2][0],\
                                detection[2][1],\
                                detection[2][2],\
                                detection[2][3],\
                                detection[1]     	# detection score
            xmin, ymin, xmax, ymax = convert_back(float(x), float(y), float(w), float(h))
            centroids[objectId] = (int(x), int(y), xmin, ymin, xmax, ymax, score)
            objectId += 1
    else:
        centroids['NaN'] = tuple() # add empty entry when no detections in a frame to keep the count
    all_detections.append(centroids)


def find_inner_bbox(detections, img, proportion):
    orig_img = img.copy()
    centroids = dict()
    if len(detections) > 0:
        factor = 0.5*(1.0 - proportion)
        objectId = 0
        for detection in detections:               
            x, y, w, h, score = detection[2][0],\
                        detection[2][1],\
                        detection[2][2],\
                        detection[2][3],\
                        detection[1]  
            xmin, ymin, xmax, ymax = convert_back(float(x), float(y), float(w), float(h))
            bbox_w = xmax-xmin
            bbox_h = ymax-ymin
            inner_xmin, inner_xmax = int(xmin + factor*bbox_w), int(xmax - factor*bbox_w)
            inner_ymin, inner_ymax = int(ymin + factor*bbox_h), int(ymax - factor*bbox_h)
            centroids[objectId] = (int(x), int(y), xmin, ymin, xmax, ymax, score, inner_xmin, inner_ymin, inner_xmax, inner_ymax)
            objectId += 1 # new id for next detection

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            line_thickness = 1

            for idx, box in centroids.items():
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 0, 255), 1) # Create blue bounding boxes
                cv2.rectangle(img, (box[7], box[8]), (box[9], box[10]), (255, 0, 0), 1) # Create red inner bounding boxes
                text_width, text_height = cv2.getTextSize(str(idx), font, font_scale, line_thickness)[0]
                box_coords = ((box[2], box[3]), (box[2] + text_width + 2, box[3] - text_height - 2))
                cv2.rectangle(img, box_coords[0], box_coords[1], (0, 0, 255), cv2.FILLED)
                cv2.putText(img, str(idx), (box[2], box[3]), font, font_scale, (255, 255, 255), line_thickness)

    h,w,_ = orig_img.shape
    left_frame = orig_img[:, :w//2, :]
    right_frame = orig_img[:, w//2:, :]
    calibration_file = "calibration_params.npz"
    rect1, rect2 = rectify(calibration_file, left_frame, right_frame)
    disp1, disp2, filtered_disp, num_disp, min_disp = disparity(rect1, rect2, filter=True)
    dm = (filtered_disp-min_disp)/num_disp

    # Save img temporary to file
    cv2.imwrite("dm.png", dm)
    
    plt.imsave("img_orig.png", orig_img[:, :w//2, :]) 
    plt.imsave("img_w_bboxes.png", img[:, :w//2, :])
    plt.close('all')

    return img, dm, centroids

def calculate_disparity_and_variance_in_bbox(bbox_mat):
    """
    bbox_mat: matrix with disparity values
    Check the variation of disparity in a small area (~10% or 10x10 px) in the middle of the bbox.
    If the variation is larger than a certain value (what value? 2*std?), then discard the disparity estimation and return None.
    Else, return the average depth value over all pixels in the bbox.
    """
    # Computing the variance of a 10% inner area of the bbox
    proportion = 0.1 # 10%
    min_size = 10 # pixels

    h,w = bbox_mat.shape
    inner_w = int(max(w*proportion, min_size))
    inner_h = int(max(h*proportion, min_size))
    start_w = int(w*(1-proportion)//2)
    start_h = int(h*(1-proportion)//2)

    inner_bbox_mat = bbox_mat[start_h:start_h+inner_h, start_w:start_w+inner_w]

    avg = inner_bbox_mat.sum()/(inner_w*inner_h)
    x = abs(inner_bbox_mat - inner_bbox_mat.mean())**2
    var = x.sum()/(inner_w*inner_h)

    # min and max variance must be tuned
    min_var = 0.000001
    max_var = 0.1

    return (avg if min_var <= var or var <= max_var else None), var


def replace_frame_with_dm(img):
    h,w,_ = img.shape
    left_frame = img[:, :w//2, :]
    right_frame = img[:, w//2:, :]
    calibration_file = "calibration_params.npz"
    rect1, rect2 = rectify(calibration_file, left_frame, right_frame)
    disp1, disp2, filtered_disp, num_disp, min_disp = disparity(rect1, rect2, filter=True)
    dm = (filtered_disp-min_disp)/num_disp
    plt.imsave("dm.png", dm, cmap=plt.cm.gray) 
    return dm

def replace_bbox_with_depthmap(dm_path, img_path, centroids, dm_values):
    """
        Replace inner bboxes with depth map.
        img[x:y, z:t] = depthmap[x:y, z:t]
    """

    img =  Image.open(img_path)
    dm = Image.open(dm_path)
    w, h = img.size
    img = img.resize((w//2, h//2))

    img2 = cv2.imread(img_path)
    h,w,_ = img2.shape
    img2 = cv2.resize(img2, (w//2,h//2))
    dm2 = cv2.merge((dm_values*255, dm_values*255, dm_values*255))

    reshaped_dm_values = dm_values.reshape((h//2, w//2))

    # get pixel information
    new_centroids = dict()

    # go through each bbox, and go through each pixel
    for idx, bbox in centroids.items():
        # normal bboxes
        bbox_2 = bbox[2]//2
        bbox_3 = bbox[3]//2
        bbox_4 = bbox[4]//2
        bbox_5 = bbox[5]//2

        bbox_2 = max(bbox_2,0)
        bbox_4 = min(bbox_4,w//2-1)
        bbox_3 = max(bbox_3,0)
        bbox_5 = min(bbox_5,h//2-1)

        avg, var = calculate_disparity_and_variance_in_bbox(reshaped_dm_values[bbox_3:bbox_5, bbox_2:bbox_4])
        new_bbox = (bbox[0], bbox[1], bbox_2, bbox_3, bbox_4, bbox_5, bbox[6], bbox[7], bbox[8], bbox[9], bbox[10], avg, var)

        # add updated bbox to new centroids
        new_centroids[idx] = new_bbox

        mask = dm2[bbox_3:bbox_5, bbox_2:bbox_4]
        
        img2[bbox_3:bbox_5, bbox_2:bbox_4] = mask
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        line_thickness = 1
        text_width, text_height = cv2.getTextSize(str(avg), font, font_scale, line_thickness)[0]
        cv2.putText(img2,"avg: " + str(avg),(bbox_2,bbox_3+text_height), font, font_scale,(255,255,255),line_thickness)
        cv2.putText(img2,"var: " + str(var),(bbox_2,bbox_3+3*text_height), font, font_scale,(255,255,255),line_thickness)

    all_detections_depth.append(new_centroids)
    return img2

def draw_tracking_boxes(detections, img):
    """
    :param:
        detections = total detections in one frame
        img = image from detect_image method of darknet
    :return:
        img with bbox
    """
    if not detections:
        return img
    
    if len(detections) > 0:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        line_thickness = 1

        for box in detections['bboxes']:
            bbox = box['bbox']
            color = box['color']
            trackid = box['id']
            cv2.rectangle(img, (bbox[2], bbox[3]), (bbox[4], bbox[5]), color, 1)
            text_width, text_height = cv2.getTextSize(str(trackid), font, font_scale, line_thickness)[0]
            box_coords = ((bbox[2], bbox[3]), (bbox[2] + text_width + 2, bbox[3] - text_height - 2))
            cv2.rectangle(img, box_coords[0], box_coords[1], color, cv2.FILLED)
            cv2.putText(img, str(trackid), (bbox[2], bbox[3]), font, font_scale, (255, 255, 255), line_thickness)
        
    return img

netMain = None
metaMain = None
altNames = None
all_detections = None
all_detections_depth = None
unique_objects = None

def pipeline(input_video_path, output_video_path, track_output_path):
    global metaMain, netMain, altNames, all_detections, all_detections_depth, unique_objects
    basePath = "Y:/Yolo_v4/darknet/build/darknet/x64"
    configPath = basePath + "/cfg/yolov4-fishfins3.cfg"
    weightPath = basePath + "/backup/fishfins3/yolov4-fishfins3_best.weights"
    metaPath = basePath + "/data/fishfins3.data"

    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.grouxp(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass

    if all_detections is None:
        all_detections = []
    if all_detections_depth is None:
        all_detections_depth = []
    if unique_objects is None:
        unique_objects = []
    
    cap = cv2.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter(
            output_video_path, cv2.VideoWriter_fourcc(*"MP4V"), 10.0,
            (frame_width // 4, frame_height // 2))
    
    print("--------------- YOLO detection started")

    # create an image to reuse for each detection
    darknet_image = darknet.make_image(frame_width // 2, frame_height, 3) # frame_width // 2 because the input image is left merged with right img

    i = 0
    # DETECTION
    while True:
        ret, frame_read = cap.read()      
        if not ret: # ret returns True if a frame was retrieved
            break
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (frame_width, frame_height), 
                                   interpolation=cv2.INTER_LINEAR)
        frame_resized_darknet = frame_resized[:, :frame_width//2, :]
        darknet.copy_image_from_bytes(darknet_image,frame_resized_darknet.tobytes())
        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        extract_coords(detections)
        image, dm_values, centroids = find_inner_bbox(detections, frame_resized, 0.15)

        # input image w/ bbox
        #image_with_depth = replace_bbox_with_depthmap("dm.png", "img_w_bboxes.png", centroids, dm_values)

        # input image w/o bbox
        image_with_depth = replace_bbox_with_depthmap("dm.png", "img_orig.png", centroids, dm_values)

        # convert images from PIL to OpenCV
        image_with_depth = cv2.cvtColor(np.array(image_with_depth), cv2.COLOR_RGB2BGR)
        cv2.imshow('Demo', image_with_depth)
        cv2.waitKey(10)
        out.write(image_with_depth)
        i += 1
    
    cap.release()
    out.release()
    print("--------------- Detection completed")

    # TRACKING
    print("--------------- Tracking started")
    start_time = time.time()
    tracks = tracker_iou_allow_gaps(all_detections_depth, 0.3, 0.3, 0.7, 10, 10)
    
    # write tracks to file
    shelf = shelve.open("tracks.slhf")
    shelf["tracks"] = tracks
    shelf.close()

    end_time = time.time()
    total_time = end_time - start_time

    tracks_by_frame = convert_tracks(tracks, total_frames)
    
    cap = cv2.VideoCapture(output_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter(
            track_output_path, cv2.VideoWriter_fourcc(*"MP4V"), 10,
            (frame_width, frame_height))

    # image to reuse for each track
    darknet_image = darknet.make_image(frame_width, frame_height, 3)

    frame_num = 0
    while True:
        frame_num += 1
        ret, frame_read = cap.read()
        if not ret:
            break

        tracks_for_frame = next((item for item in tracks_by_frame if item['frame'] == frame_num), False)

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (frame_width, frame_height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
        image = draw_tracking_boxes(tracks_for_frame, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow('Tracking', image)
        cv2.waitKey(3)
        out.write(image)

    cap.release()
    out.release()
    print("Video write of tracking result completed")
    print("Tracking completed in", total_time, "s, on", total_frames, "frames.")
    print("Total tracks obtained:", len(tracks))
    max_t = max(tracks, key=lambda t: len(t['bboxes']))
    print("Maximum track length:", len(max_t['bboxes']))

    tot_len = 0
    for t in tracks:
        tot_len += len(t['bboxes'])
    
    print("Average track length:", tot_len/len(tracks))

if __name__ == "__main__":
    output_video_path = "output.avi"
    input_path_merged = "input.mp4"
    track_output_path = "track_output.avi"
    pipeline(input_path_merged, output_video_path, track_output_path)
