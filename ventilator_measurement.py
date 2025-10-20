import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

ROOT_PATH = Path(__file__).parent.resolve().absolute()
# Adjust path to image as needed
# IMAGE_PATH = ROOT_PATH / "test images(from google)/ventilator-with-fan-window.jpg"  
IMAGE_PATH = ROOT_PATH / "test images(from google)/bathroom-fan-installation.jpg"


PERFECT_CAMERA = False   # True: only 2-point calibration, False: perspective transformation with 4 points

# Select two points in the image whose real-world distance you know.
def select_points_and_calculate_angle(img):
    print("Click two points in the image whose real-world distance you know.")
    points = []
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
            cv2.imshow('Select 2 Points', img)
    temp_img = img.copy()
    cv2.imshow('Select 2 Points', temp_img)
    cv2.setMouseCallback('Select 2 Points', mouse_callback)
    while len(points) < 2:
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    pixel_dist = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
    real_dist_mm = float(input('Enter the real distance between the points in mm: '))
    # Calculate angle (in degrees) between the line and the horizontal axis5
    dx = points[1][0] - points[0][0]
    dy = points[1][1] - points[0][1]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    print(f"Angle between points (relative to horizontal): {angle_deg:.2f} degrees")
    print(f"Pixel distance: {pixel_dist:.2f} px, Real distance: {real_dist_mm:.2f} mm")
    scale = real_dist_mm / pixel_dist
    return scale, angle_deg, points

def perspective_transform_from_points(img):
    print("Click four points in the image (in order: top-left, top-right, bottom-right, bottom-left)")
    points = []
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Select 4 Points', img)
    temp_img = img.copy()
    cv2.namedWindow('Select 4 Points', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Select 4 Points', 1000, 600)
    cv2.imshow('Select 4 Points', temp_img)
    cv2.setMouseCallback('Select 4 Points', mouse_callback)
    while len(points) < 4:
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    real_width = float(input('Enter real width in mm: '))
    real_height = float(input('Enter real height in mm: '))
    pts_src = np.array(points, dtype='float32')
    pts_dst = np.array([[0,0],[real_width,0],[real_width,real_height],[0,real_height]], dtype='float32')
    max_width = int(real_width)
    max_height = int(real_height)
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    img_warped = rewarpping_image(pts_src, pts_dst, img, output_site_px=500)
    cv2.imshow('img_warped', img_warped)
    return img_warped, real_width, real_height

def rewarpping_image(pts_src, pts_dst, image, output_site_px=500):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    H, _ = cv2.findHomography(pts_src, pts_dst) 
    # Estimate the bounding box of the warped image
    height, width = image_rgb.shape[:2]
    corners = np.array([[0,0], [width,0], [width,height], [0,height]], dtype=np.float32).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners, H)
    x_coords = warped_corners[:,0,0]
    y_coords = warped_corners[:,0,1]

    min_x, max_x = int(np.floor(x_coords.min())), int(np.ceil(x_coords.max()))
    min_y, max_y = int(np.floor(y_coords.min())), int(np.ceil(y_coords.max()))

    # Compute new canvas size
    canvas_width = max_x - min_x
    canvas_height = max_y - min_y

    # Adjust homography to shift image into positive coordinates
    translation_matrix = np.array([[1, 0, -min_x],
                                [0, 1, -min_y],
                                [0, 0, 1]], dtype=np.float32)
    H_translated = translation_matrix @ H

    # Create white canvas and warp image onto it
    white_canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    warped = cv2.warpPerspective(image_rgb, H_translated, (canvas_width, canvas_height), dst=white_canvas, borderMode=cv2.BORDER_TRANSPARENT)
    
    return warped

img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"Error: Image '{IMAGE_PATH}' could not be loaded.")
    exit(1)
# Resize image
resize_factor = 0.2
img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)

if PERFECT_CAMERA:
    # Only 2-point calibration, no rectification
    scale, angle_deg, ref_points = select_points_and_calculate_angle(img)
else:
    # Perspective transformation with 4 points
    img, real_width, real_height = perspective_transform_from_points(img)
    scale, angle_deg, ref_points = select_points_and_calculate_angle(img)

# Remove contour detection and drawing
img_contour = img.copy()

# Interactive selection of two points and display of distance in mm
print("Select any two points in the image to measure the distance (all measurements in mm).")
selected_points = []
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert pixel coordinates to mm
        x_mm = x * scale
        y_mm = y * scale
        selected_points.append((x_mm, y_mm))
        # Marking in the image (optional, still at pixel position)
        cv2.circle(img_contour, (x, y), 7, (0, 0, 255), -1)
        cv2.imshow('Measure Distance', img_contour)

img_temp = img_contour.copy()
cv2.imshow('Measure Distance', img_temp)
cv2.setMouseCallback('Measure Distance', mouse_callback)
while len(selected_points) < 2:
    cv2.waitKey(1)
cv2.destroyAllWindows()

# Calculate and display distance (in mm)
p1 = np.array(selected_points[0])
p2 = np.array(selected_points[1])
dist_mm = np.linalg.norm(p1 - p2)
print(f"Distance between the selected points: {dist_mm:.2f} mm")

# Draw line and text (still at pixel basis, but text in mm)
pixel_p1 = (int(p1[0]/scale), int(p1[1]/scale))
pixel_p2 = (int(p2[0]/scale), int(p2[1]/scale))
cv2.line(img_contour, pixel_p1, pixel_p2, (0,0,255), 2)
cv2.putText(img_contour, f"{dist_mm:.1f} mm", ((pixel_p1[0]+pixel_p2[0])//2, (pixel_p1[1]+pixel_p2[1])//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
cv2.imshow('Measured Distance', img_contour)
cv2.waitKey(3000)
cv2.destroyAllWindows()

# Interactive selection for wall distances
for wall_name in ["left wall", "right wall"]:
    print(f"Select two points for the distance to the {wall_name}.")
    wall_points = []
    def wall_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x_mm = x * scale
            y_mm = y * scale
            wall_points.append((x_mm, y_mm))
            cv2.circle(img_contour, (x, y), 7, (0, 255, 0), -1)
            cv2.imshow(f'Measure Distance to {wall_name}', img_contour)
    img_temp = img_contour.copy()
    cv2.imshow(f'Measure Distance to {wall_name}', img_temp)
    cv2.setMouseCallback(f'Measure Distance to {wall_name}', wall_callback)
    while len(wall_points) < 2:
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    wp1 = np.array(wall_points[0])
    wp2 = np.array(wall_points[1])
    wall_dist_mm = np.linalg.norm(wp1 - wp2)
    print(f"Distance to {wall_name}: {wall_dist_mm:.2f} mm")
    pixel_wp1 = (int(wp1[0]/scale), int(wp1[1]/scale))
    pixel_wp2 = (int(wp2[0]/scale), int(wp2[1]/scale))
    cv2.line(img_contour, pixel_wp1, pixel_wp2, (0,255,0), 2)
    cv2.putText(img_contour, f"{wall_dist_mm:.1f} mm", ((pixel_wp1[0]+pixel_wp2[0])//2, (pixel_wp1[1]+pixel_wp2[1])//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow('Measured wall distances', img_contour)
    cv2.waitKey(0) #quit on key press
    cv2.destroyAllWindows()
