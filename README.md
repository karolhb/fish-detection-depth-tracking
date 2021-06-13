# Vision-Based Detection, Depth Estimation, and Tracking of Fish
Repository for work conducted in the course _TMR4930 - Marine Technology, Master's Thesis_.

The project uses YOLOv4 for object detection, ?? for depth estimation, and a simple IOU tracking algorithm based on [E. Bochinski, V. Eiselein and T. Sikora, 2017](https://ieeexplore.ieee.org/document/8078516). First, fish tails fins are detected. Depth information in the bounding boxes is estimated, and averaged over a small patch in the middle of the bounding box (likely to cover only fin surface). The tail fins are further tracked in consecutive frames. The depth information combined with the tracks will be used for extracting information about swimming velocity of the fish.

## Dependencies
_TODO: Add versions._
- CUDA
- Darknet
- OpenCV
- FFmpeg
- Python PIL
- Tensorflow
- Keras
- Pandas

## Repository Description
_TODO: Add remaining code_
- Utils: Contains a variety of utility functions used for tasks like file processing (rename files, move files, etc.), image processing (merge left and right image, etc.), and video processing (concatenate video streams, extract frames from video, etc.).

## Project Description
Monitoring farmed fish welfare is crucial from ethical and financial perspectives. Going towards a more autonomous fish farming industry, there are many aspects that can be explored using computer vision-based techniques on camera data from farmed fish sea cages to monitor fish behaviour. Deep-learning based computer vision enables adaptable, scalable, and data-driven methods suitable for overcoming challenges of the dynamic underwater environment in sea cages. Examples of indicators of fish welfare that can be captured by vision are swimming speed, lice, wounds/injuries, gill cover frequency, and fish orientation. This master’s thesis will thus investigate vision-based techniques on stereo camera data from farmed salmon sea cages to monitor swimming behavior of farmed salmon.

The thesis is written in collaboration with FiiZK, a supplier of software, closed cage, and technical tarpaulins in the aquaculture industry, and SINTEF, a research organization that conducts contract research and development projects. Through FiiZK, the necessary equipment for running training of machine learning models was provided. SINTEF provided stereo camera data from salmon sea cages. Investigation of data availability and evaluation of the limitation of the data was a part of the work in the thesis.

__Workflow__
- Data acquisition and decision of main focus of the thesis: Investigate available camera data from salmon sea cages and record new data if needed. The data availability and limitations will be evaluated to formulate the scope of the thesis.
- Perform a background and literature review on:
  * Fish welfare indicators based on fish behaviour.
  * Previous studies on monitoring fish behaviour.
  * Relevant traditional computer vision methods.
  * Relevant (deep-learning) vision-based techniques.
- Choose fish welfare indicators to look closer at, by:
  * Classification of indicators by visual and temporal detectability.
  * Investigation of the quality and size of the dataset.
- Computer vision and deep learning tasks:
  * Set up computer and working environment(s).
  * Prepare and annotate datasets.
  * Perform tasks using chosen vision-based techniques on the prepared dataset.
  * Validate the method(s).
- Evaluate the results:
  * Accuracy of object detection.
  * Point clouds generated with depth estimation model.
  * Track lengths and number of tracks detected by the tracking algorithm.
  * Total pipeline.
- Make a recommendation for applications, improvements, and further work.
