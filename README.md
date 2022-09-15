# Developing a Smart Video Surveillance System for Violence Detection and Aggressor Identification Using Pose Estimation and Support Vector Machines
by: Paolo Sandejas, Jomari Deligero 
Computer Vision and Machine Learning Group - University of the Philippines Diliman

## Abstract

Automatically detecting acts of violence and identifying violent individuals in footage captured by CCTVs is an essential next step for these systems to better ensure a more secure and safe society. Our research aims to develop an efficient near real-time video surveillance system to detect violence and identify aggressors in urban areas using Pose Estimation and Support Vector Machines (SVM). Our computer vision framework takes features (velocity, angle, suspicious contact) extracted from the key point output of a pose estimation algorithm, OpenPose, and feeds that data into an SVM for violence detection. While our research mainly focuses on evaluating actions between two people, this study also adds a novel feature that identifies the aggressor in a violent scene given the amount of suspicious contact detected. Different versions of the model were tested and the best performing model achieved great results on all metrics.

## Link to Papers
For more information about our project please access the following drive link: https://drive.google.com/drive/folders/10SZvZnyHvbaZscJZ_POKgL8tn8mvPDVO?usp=sharing

## Set-Up and Installation
### Prerequisites
In order to run our code, OpenPose must first be successfully installed along with its python API. To do this, please refer to OpenPose's github repo linked here: 
- About OpenPose: https://github.com/CMU-Perceptual-Computing-Lab/openpose#installation
- OpenPose Installation Guide: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md#compiling-and-running-openpose-from-source
- OpenPose Python API: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/03_python_api.md

Remember to follow the steps in order to run OpenPose in Python

### Obtaining the datasets
The ISR-UoL 3D Social Activity Dataset was used to train our model on violent and non-violent actions. 

The dataset can be found here: https://lcas.lincoln.ac.uk/wp/research/data-sets-software/isr-uol-3d-social-activity-dataset/#:~:text=This%20is%20a%20social%20interaction,by%20an%20RGB%2DD%20sensor.

This video dataset was translated into a CSV file after extracting the our chosen features from each video frame and storing these features as a row in our CSV frame dataset.

Our frame datasets can be found here: https://drive.google.com/drive/folders/1DUTkCno2_vR4YFwHQVHlFA7DJ-1cQpnR?usp=sharing


### Running our code
Once OpenPose has been successfully installed, place the contents of this repo and the frame_datain the following directory inside the openpose repo
```
build/examples/tutorial_api_python
```
This folder should already exist if the OpenPose python API was setup properly. Afterwards, change the default image directory accessed in violence_detection-v5-twenty_frames.py to the directory containing the video frames you wish to analyze and run the Python file
```
python violence_detection-v5-twenty_frames.py
```


