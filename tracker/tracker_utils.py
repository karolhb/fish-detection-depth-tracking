# -------------------------------------------------------------------
# Various utility functions
# Written by Karoline H. Barstein
# -------------------------------------------------------------------

import random
import math

def rand_col():
  return random.randint(0,255)

def convert_back(x, y, w, h): 
    """
    :param:
        x, y = midpoint of bbox
        w, h = width, height of the bbox
    
    :return:
        xmin, ymin, xmax, ymax
    """
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def calc_iou(bb1, bb2):
    """
    :params:
        bb1, bb2 = bounding boxes (xcenter, ycenter, xmin, ymin, xmax, ymax, score)
    :return:
        iou score
    """
    assert bb1[2] <= bb1[4]
    assert bb1[3] <= bb1[5]
    assert bb2[2] <= bb2[4]
    assert bb2[3] <= bb2[5]

    x_left = max(bb1[2], bb2[2])
    y_top = max(bb1[3], bb2[3])
    x_right = min(bb1[4], bb2[4])
    y_bottom = min(bb1[5], bb2[5])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1[4] - bb1[2]) * (bb1[5] - bb1[3])
    bb2_area = (bb2[4] - bb2[2]) * (bb2[5] - bb2[3])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def convert_tracks(tracks, total_frames):
    tracks_by_frame = []

    for id, track in enumerate(tracks, start=0):
        color = (rand_col(),rand_col(),rand_col())
        for idx, bbox in enumerate(track['bboxes']):
            frame = track['start_frame'] + idx
            frame_entry = next((track_frame for track_frame in tracks_by_frame if track_frame["frame"] == frame), False)
            if frame_entry: # if frame already has bboxes
                frame_entry['bboxes'].append({'bbox': bbox, 'id': id, 'color': color})
            else:
                tracks_by_frame.append({'frame': frame, 'bboxes': [{'bbox': bbox, 'id': id, 'color': color}]})

    return tracks_by_frame

def get_tracks_avg_velocities(tracks):
    """
    Calculate average velocity of each track's center point.
    :params:
        track = list of tracks
    :return:
        average velocities for all tracks
    """
    for track in tracks:
        total_dist = 0
        if len(track['bboxes']) > 1:
            for i in range(1, len(track['bboxes'])):
                bb1 = track['bboxes'][i-1]
                bb2 = track['bboxes'][i]
                total_dist += math.sqrt((bb2[0] - bb1[0])**2 + (bb2[1] - bb1[1])**2) # pixel distance from frame i-1 to frame i
        track_vel = total_dist/len(track['bboxes']) if len(track['bboxes']) > 0 else 0
        track['vel'] = track_vel

    return tracks





