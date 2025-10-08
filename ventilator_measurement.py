import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "test_pic.png"  # Pfad zum Bild anpassen

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
    # Calculate angle (in degrees) between the line and the horizontal axis
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
    img_warped = cv2.warpPerspective(img, M, (max_width, max_height))
    return img_warped, real_width, real_height

img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"Fehler: Bild '{IMAGE_PATH}' konnte nicht geladen werden.")
    exit(1)
    
# Bild verkleinern
resize_factor = 0.1
img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)

# Apply perspective transform
img, real_width, real_height = perspective_transform_from_points(img)

# Select two points and calculate angle BEFORE any processing or plotting
scale, angle_deg, ref_points = select_points_and_calculate_angle(img)

# Vorverarbeitung
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (11, 11), 0) ## weich zeichnen 
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

cv2.imshow('thres', thresh)
cv2.waitKey(300)  # Show threshold image for 300 ms
cv2.destroyAllWindows()

contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

# Create a copy of the image for plotting
img_contour = img.copy()

# Only show and label the largest 10 contours for speed
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

for i, contour in enumerate(contours):
    length_px = cv2.arcLength(contour, closed=True)
    length_mm = length_px * scale
    print(f"Contour {i}: Length = {length_mm:.2f} mm")
    cv2.drawContours(img_contour, [contour], -1, (0, 255, 0), 2)
    x, y = contour[0][0]
    cv2.putText(img_contour, f"{length_mm:.1f} mm", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

cv2.imshow('Contours with Lengths', img_contour)
cv2.waitKey(3000)  # Show result for 1 second
cv2.destroyAllWindows()
exit(0)