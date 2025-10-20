# FanVision: 3D Measurement System for Fan Installation - IDS Computer Vision Hackathon
This repository contains code and resources for the Fan Installation Vision Measurement project, developed as part of the IDS Computer Vision Hackathon. The project focuses on using computer vision techniques to measure and analyze fan installations.
## Hackathon Team Members:



## Introduction
FanVision is a computer vision–based measurement system that enables **accurate spatial mapping** of fans and their surroundings using **3D camera input**. 
> ❗ **Note:** For this, it requires the company's 3D camera. Those 3D photos we captured during Hackathon are **confidential** and **cannot** be shared here.


For the sake of the users' convenience, we built this system to support **mobile device input**. 
(But note that, the ventilator photos we demenstrate here are not the actual ones we used during the Hackathon due to the **security concerns**, but similar ones we created for demonstration purposes)
> ❗ **Note:** the ventilator photos we demenstrate here are **Not** the actual ones we used during the Hackathon due to the **security concerns**, but similar ones we replaced for demonstration purposes 


It helps ensure **optimal installation distance** for airflow efficiency and safety in ventilation systems.

## Challenge Description
1. **measure the ventilator's surrounding physical dimensions using image and depth data.**  
2. **Measure the gap between the ventilator frame and the spinning fan blade.**  

## Solution Structure

### 1. Data Acquisition
Capture the photos of the ventilator and its surroundings using a 3D camera (from the company, usually it takes some time on delivery) or mobile device (for the convenience for the customers).

### 2. Preprocessing
#### Edge Detection and Enhancement


#### Image Skewing




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


### Here is an example:

#### Original Image(impefectly aligned):
![Image](https://github.com/user-attachments/assets/8c1e25c2-a5a2-4587-8ad6-5028f7ba3e5f)


#### After Preprocessing:
<img width="1156" height="1174" alt="Image" src="https://github.com/user-attachments/assets/525bd767-a3b3-4d6e-a8d5-cf17d5386c7d" />

> ❗ **Note again:** the above image is not the initial image we used during Hackathon, but a similar one for demonstration purposes.


