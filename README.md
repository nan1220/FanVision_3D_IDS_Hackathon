# FanVision: 3D Measurement System for Fan Installation - IDS Computer Vision Hackathon
This repository contains code and resources for the Fan Installation Vision Measurement project, developed as part of the IDS Computer Vision Hackathon (08.10.2025-09.10.2025). The project focuses on using computer vision techniques to measure and analyze fan installations.

> ❗ **Note:** The project was finished during the Hackathon event (08.10.2025-09.10.2025), but due to the confidentiality agreement with the company, some confidential data and details cannot be shared here. I have replaced those confidential parts with similar data for demonstration purposes after the Hackathon.
## Hackathon Team Members:

- **[Nan Jiang](https://www.linkedin.com/in/nan-jiang-tum)**   <br>

- **[Bavly Kirolos](https://www.linkedin.com/in/bavly-kirolos-01a45321b)** <br>
- **[Kassandra Karger](linkedin.com/in/kassandra-karger)**<br>
- **[Louis Wolf](https://www.linkedin.com/in/louis-wolf-686848108)** <br>
- **[Felix Kammerer](linkedin.com/in/felix-kammerer-4a7848170)** <br>



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

### Here is an example of our solution by video demonstration:

https://github.com/user-attachments/assets/454403b8-ff97-4158-9227-d3e494e39b83

> ❗ **Note:** the above image is not the initial image we used during Hackathon, but a similar one for demonstration purposes. I replaced the confidential photos and recorded it after Hackathon to demonstrate the solution.

## Solution Structure

### 1. Data Acquisition
Capture the photos of the ventilator and its surroundings using a 3D camera (from the company, usually it takes some time on delivery) or mobile device (for the convenience for the customers).

### 2. Preprocessing(Image Skewing)

To correct perspective distortion and obtain a front-facing (rectangle) view of the ventilator, a homography-based perspective transformation was applied.

#### Input: Original image with imperfectly aligned ventilator frame.

#### Process:

Use interactive point selection to mark the four corners of the ventilator frame.

Define the coordinates of the selected corner points in the input image.

Define corresponding destination points forming a perfect rectangle.

Compute the homography matrix using `cv2.getPerspectiveTransform()`.

Apply the transformation with `cv2.warpPerspective()` to produce a rectified image.

#### Output: 
The ventilator frame becomes a perfect rectangle, reducing geometric distortion and enabling more accurate edge and shape detection in later steps.

#### Example:
> ❗ **Note again:** this is not the photo we used during Hackathon, but a similar one for demonstration purposes


| Input: Original Image(impefectly aligned)| Output: Warped (rectified) image |
| ----------- | ----------- |
| ![Image](https://github.com/user-attachments/assets/8c1e25c2-a5a2-4587-8ad6-5028f7ba3e5f)| ![Image](https://github.com/user-attachments/assets/6e9b8e08-2342-4ed4-91bf-424b6129903c)







### 3. Object Detection
- Detect fan blades and ventilator frames using:
  - **Hough Circle Transform** for circular blades

  for example:

  <img width="431" height="388" alt="Image" src="https://github.com/user-attachments/assets/2f410121-abde-47d1-988f-674cc72ecc40" />

  - **Contour and edge analysis** for rectangular ventilator frames
- Label detected objects with bounding boxes or contours.

### 4. Depth & Distance Estimation
#### For skewed images:
we would first use the image preprocessing and skew the image to make the lines percfectly aligned, then estimate the dimensions of the surroundings based on the skewed image and known data, e.g., width and height of the ventilator frame, diameter of the fan blades, etc.

##### Example of estimating dimensions from skewed image:
<img width="1156" height="1174" alt="Image" src="https://github.com/user-attachments/assets/525bd767-a3b3-4d6e-a8d5-cf17d5386c7d" />

(there is less than 1cm error.)
> ❗ **Note again:** the above image is not the initial image we used during Hackathon, but a similar one for demonstration purposes.

#### For perfectly aligned images:
We don't need to skew the image. We can directly use the perfectly aligned image to estimate the dimensions of the surroundings.



