import ffmpeg
import os
from os import listdir
import random
import cv2 as cv

def trim_by_frame(in_file, out_file, start_frame, num_frames):
    """
    :params:
        in_file = video to be trimmed
        out_file = filename for output video
        start_frame = frame number to start trim
        num_frames = number of frames in trimmed video
    """
    (
        ffmpeg
        .input(in_file)
        .trim(start_frame=start_frame, end_frame=start_frame+num_frames)
        .output(out_file)
        .run()
    )

def trim_by_time(in_file, out_file, start_time, num_sec):
    """
    :params:
        in_file = video to be trimmed
        out_file = filename for output video
        start = time (sec) to start trim
        num_sec = number of sec in trimmed video
    """
    (
        ffmpeg
        .input(in_file)
        .trim(start=start_time, end=start_time+num_sec)
        .output(out_file)
        .run()
    )

def concatenate_videos(left_vid, right_vid, output_file_path):
    # use ffmpeg
    # make this one work, and use the merged video in the depth map (not detection? )
    inputs = [left_vid, right_vid]
    in_left = ffmpeg.input(left_vid)
    in_right = ffmpeg.input(right_vid)
    out = ffmpeg.filter(
            [in_left, in_right], 'hstack'
        ).output(output_file_path)

    out = ffmpeg.overwrite_output(out)
    out.run()
