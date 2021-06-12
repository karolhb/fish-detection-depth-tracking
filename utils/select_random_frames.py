import ffmpeg
import os
from os import listdir
import random
import cv2 as cv

"""
Choose, cut, and save a number random frames from a collection of videos.
"""

def get_frame(in_fn, out_fn, frame):
    """
    :params:
        in_fn = filename for input video
        out_fn = filename for output frame
        frame = frame number to extract
    """
    (
        ffmpeg
        .input(in_fn)
        .trim(start_frame=frame, end_frame=frame+1)
        .output(out_fn)
        .run()
    )

def concat_videos(video_files: list, in_folder, out_name, fps=14, h=696, w=568):
    """
    :params:
        video_files = list of files to concat
        in_folder = folder for input videos
        out_name = filename for output video
        fps = frames per sec for output video
        h, w = min height and width of videos to be concatenated
    """
    if len(video_files) > 1:
        inputs = [ffmpeg.input(in_folder+vid).crop(0,0,h,w) for vid in video_files]
        out = ffmpeg.concat(
            *inputs
        ).filter('mpdecimate').filter('fps', fps=fps, round='up').output(out_folder+joined_name)

        out = ffmpeg.overwrite_output(out)
        out.run()

def get_frame_count(filename):
    """
    :params:
        filename = name of video file
    """
    cap = cv.VideoCapture(filename)
    return int(cap.get(cv.CAP_PROP_FRAME_COUNT))

def get_random_sample(min, max, count):
    """
    :params:
        min = random range min
        max = random range max
        count = number of random numbers to return
    """
    return random.sample(range(min, max), count)

def select_random_frames(in_folder, out_folder, joined_name, out_filename, out_filetype, samples):
    """
    :params:
        in_folder = folder with input videos
        out_folder = folder to save output video in
        joined_name = name of concatenated video file
        out_filename = name of extracted image frames
        out_filetype = filetype of extracted image frames
        samples = number of random frames
    """
    video_files = os.listdir(in_folder)
    concat_videos(video_files, in_folder, joined_name)
    no_of_frames = get_frame_count(out_folder+joined_name)
    random_frames = get_random_sample(1, no_of_frames, no_of_samples)
    count = 0
    for frame in random_frames:
        get_frame(out_folder+joined_name, out_folder+out_filename+str(count)+out_filetype, frame)
        count += 1
