# FanVision: 3D Measurement System for Fan Installation - IDS Computer Vision Hackathon
This repository contains code and resources for the Fan Installation Vision Measurement project, developed as part of the IDS Computer Vision Hackathon. The project focuses on using computer vision techniques to measure and analyze fan installations.
## Team Members:

## Introduction
FanVision is a computer vision–based measurement system that enables **accurate spatial mapping** of fans and their surroundings using **3D camera input**.  
It helps ensure **optimal installation distance** for airflow efficiency and safety in ventilation systems.

## Challenge Description
1. **measure the ventilator's surrounding physical dimensions using image and depth data.**  
2. **Measure the gap between the ventilator frame and the spinning fan blade.**  

## Solution Structure

### 1. Data Acquisition
Capture the photos of the ventilator and its surroundings using a 3D camera (from the company, usually it takes some time on delivery) or mobile device (for the convenience for the customers).

### 2. Preprocessing
#### Edge Detection and Enhancement
![Image](https://github.com/user-attachments/assets/e1c770e9-e306-47e3-bf5e-b180e7e704ce)

#### Image Skewing
<img width="938" height="528" alt="Image" src="https://github.com/user-attachments/assets/865f53b5-e4f0-4503-834b-59e0c13eb5cc" />



### 3. Object Detection
- Detect fan blades and ventilator frames using:
  - **Hough Circle Transform** for circular blades
  - **Contour and edge analysis** for rectangular ventilator frames
- Label detected objects with bounding boxes or contours.

### 4. Depth & Distance Estimation
#### For skewed images:
we would first use the image preprocessing and output the image. Then we would the preprocessed image to find the four corners of the skewed image. After that, we would use the four corners to calculate the homography matrix and then use the homography matrix to warp the image to a top-down view. Finally, we would use the warped image to measure the distance between the fan blades and the ventilator frame.


#### For perfectly aligned images:
We don't need to skew the image. We can directly use the perfectly aligned image to estimate the dimensions of the surroundings.






