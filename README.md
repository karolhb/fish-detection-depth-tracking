# Vision-Based Detection, Depth Estimation, and Tracking of Fish
Repository for work conducted in the course _TMR4930 - Marine Technology, Master's Thesis_.

The project uses YOLOv4 [(Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao, 2020)](https://arxiv.org/abs/2004.10934) for object detection, semi-global block matching (by [OpenCV](https://docs.opencv.org/4.5.2/d2/d85/classcv_1_1StereoSGBM.html)) for depth estimation, and a simple IOU tracking algorithm based on [E. Bochinski, V. Eiselein and T. Sikora, 2017](https://ieeexplore.ieee.org/document/8078516). First, fish tail fins are detected. Depth information in the bounding boxes is estimated, and averaged over a small patch in the middle of the bounding box (likely to cover only fin surface). The tail fins are further tracked in consecutive frames. The depth information combined with the tracks will be used for extracting information about swimming velocity of the fish.

_Remark: This repository is an attachment for my Master's thesis at NTNU._

## Dependencies
- CUDA
- Darknet
- OpenCV
- FFmpeg
- Python PIL
- Pandas

## Repository Description
- utils: Contains a variety of utility functions used for tasks like file processing (rename files, move files, etc.), image processing (merge left and right image, etc.), and video processing (concatenate video streams, extract frames from video, etc.).
- stereo_matching: Contains functionality for camera calibration and stereo matching with OpenCV.
- tracker: Contains IOU overlap based trackers. One original version and one modified version that allows "gaps" in detections between subsequent frames.
- pipeline: Contains the final pipeline combining detection, stereo matching, and tracking.

## Project Description
Monitoring farmed fish welfare is crucial from ethical and financial perspectives. Going towards a more autonomous fish farming industry, there are many aspects that can be explored using computer vision-based techniques on camera data from farmed fish sea cages to monitor fish behaviour. Deep-learning based computer vision enables adaptable, scalable, and data-driven methods suitable for overcoming challenges of the dynamic underwater environment in sea cages. Examples of indicators of fish welfare that can be captured by vision are swimming speed, lice, wounds/injuries, gill cover frequency, and fish orientation. This master’s thesis will thus investigate vision-based techniques on stereo camera data from farmed salmon sea cages to monitor swimming behavior of farmed salmon.

The thesis is written in collaboration with FiiZK, a supplier of software, closed cage, and technical tarpaulins in the aquaculture industry, and SINTEF Ocean, a research organization that conducts contract research and development projects. Through FiiZK, the necessary equipment for running machine learning models was provided. SINTEF provided stereo camera data from salmon sea cages. Investigation of data availability and evaluation of the limitation of the data was a part of the work in the thesis.
