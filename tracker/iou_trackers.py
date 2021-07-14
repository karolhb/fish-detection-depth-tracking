# -------------------------------------------------------------------
# Program for modified IOU tracking allowing gaps
# Written by Karoline H. Barstein
# Based on Erik Bochinski's IOU Tracker:
# https://github.com/bochinski/iou-tracker/blob/master/iou_tracker.py 
# -------------------------------------------------------------------

from time import time
from utils import calc_iou
import uuid

def tracker_iou(detections, sigma_l, sigma_h, sigma_iou, t_min):
    """
        detection_dict: entry is (xcenter, ycenter, xmin, ymin, xmax, ymax, score)
        sigma_l: low detection threshold (use 0.1?)
        sigma_h: high detection threshold (use 0.7?)
            => used to filter tracks s.t. they have at least 1 bbox with confidence > sigma_h
        sigma_iou: IOU threshold (use 0.5?)
        t_min: minimum frame count (use 5?)
            => a track must last for at least t_min frames
    """
    
    if not len(detections):
        return
    
    tracks_active = []
    tracks_finished = []

    for frame_num, detections_frame in enumerate(detections, start=1):
        if list(detections_frame.values()) and len(list(detections_frame.values())[0]) > 6:
            # Filter detections. Keep only detections with score above sigma_l
            dets = [det for det in list(detections_frame.values()) if det[6] >= sigma_l] # det[6] is the score

            updated_tracks = []
            for track in tracks_active:
                if len(dets) > 0:
                    # get det with highest iou
                    best_match = max(dets, key=lambda x: calc_iou(track['bboxes'][-1], x))
                    if calc_iou(track['bboxes'][-1], best_match) >= sigma_iou:
                        track['bboxes'].append(best_match)
                        track['max_score'] = max(track['max_score'], best_match[6])

                        updated_tracks.append(track)

                        # remove from best matching detection from detections
                        del dets[dets.index(best_match)]

                # if track was not updated
                if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                    tracks_finished.append(track)

            # create new tracks
            new_tracks = [{'bboxes': [det], 'max_score': det[6], 'start_frame': frame_num} for det in dets]
            tracks_active = updated_tracks + new_tracks

    # finish all remaining active tracks
    tracks_finished += tracks_active

    # filter out tracks not filling requirements
    tracks_finished = [track for track in tracks_finished
                        if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min]

    return tracks_finished


def tracker_iou_allow_gaps(detections, sigma_l, sigma_h, sigma_iou, t_min, t_gap):
    """
    detection_dict: entry is (xcenter, ycenter, xmin, ymin, xmax, ymax, score)
    sigma_l: low detection threshold
    sigma_h: high detection threshold
        => used to filter tracks s.t. they have at least 1 bbox with confidence > sigma_h
    sigma_iou: IOU threshold
    t_min: minimum frame count
        => a track must last for at least t_min frames
    t_gap: maximum gap length (# frames)
    """

    if not len(detections):
        return
    
    tracks_active = []
    tracks_finished = []
    tracks_temp = []

    for frame_num, detections_frame in enumerate(detections, start=1):
        if len(list(detections_frame.values())) > 6:
            # apply low threshold to detections
            dets = [det for det in list(detections_frame.values()) if det[6] >= sigma_l]

            updated_tracks = []
            temp_tracks = []
            for track in tracks_active:
                if len(dets) > 0:
                    # get det with highest iou
                    best_match = max(dets, key=lambda x: calc_iou(track['bboxes'][-1], x))
                    if calc_iou(track['bboxes'][-1], best_match) >= sigma_iou:
                        track['bboxes'].append(best_match)
                        track['max_score'] = max(track['max_score'], best_match[6])

                        updated_tracks.append(track)

                        # remove best matching detection from detections
                        del dets[dets.index(best_match)]

                # if track was not updated
                if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                    tracks_finished.append(track)
                    temp_track = {'bboxes': [track['bboxes'][-1]], 'max_score': track['max_score'], 'id': track['id']}
                    temp_tracks.append(temp_track)

            # loop through all temp tracks
            updated_temp = False
            for track in tracks_temp:
                if len(dets) > 0 and len(tracks_finished) > 0:
                    # get det with highest iou
                    best_match = max(dets, key=lambda x: calc_iou(track['bboxes'][-1], x))
                    if calc_iou(track['bboxes'][-1], best_match) >= sigma_iou:
                        track['bboxes'].append(best_match)
                        track['max_score'] = max(track['max_score'], best_match[6])

                        active_track = next((t for t in tracks_finished if t['id'] == track['id']), None)
                        if active_track:
                            active_track['bboxes'] += track['bboxes'][1:] # remove first element which is "fake"

                            # new max_score
                            active_track['max_score'] = max(active_track['max_score'], track['max_score'])

                            # remove from finished
                            tracks_finished = [t for t in tracks_finished if t['id'] is not track['id']]

                            # add to updated
                            updated_tracks.append(active_track)

                            # remove from best matching detection from detections
                            del dets[dets.index(best_match)]
                            updated_temp = True

                # if track was not updated
                if not updated_temp:
                    if len(track['bboxes'])-1 <= t_gap:
                        track['bboxes'].append(track['bboxes'][-1]) # add bbox identical to prev bbox
                        temp_tracks.append(track) # add the updated temp track

            # create new tracks
            tracks_temp = temp_tracks
            new_tracks = [{'bboxes': [det], 'max_score': det[6], 'start_frame': frame_num, 'id': str(uuid.uuid4())} for det in dets]
            tracks_active = updated_tracks + new_tracks

    # finish all remaining active tracks
    tracks_finished += tracks_active

    # filter out tracks not meeting requirements
    tracks_finished = [track for track in tracks_finished
                        if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min]

    return tracks_finished