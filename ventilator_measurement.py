import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

ROOT_PATH = Path(__file__).parent.resolve().absolute()
IMAGE_PATH = ROOT_PATH / "others/image4.jpg"  # Pfad zum Bild anpassen
PERFECT_CAMERA = False   # True: nur 2-Punkt-Kalibrierung, False: Perspektivtransformation mit 4 Punkten

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
    print(f"Fehler: Bild '{IMAGE_PATH}' konnte nicht geladen werden.")
    exit(1)
# Bild verkleinern
resize_factor = 0.2
img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)

if PERFECT_CAMERA:
    # Nur 2-Punkt-Kalibrierung, keine Entzerrung
    scale, angle_deg, ref_points = select_points_and_calculate_angle(img)
else:
    # Perspektivtransformation mit 4 Punkten
    img, real_width, real_height = perspective_transform_from_points(img)
    scale, angle_deg, ref_points = select_points_and_calculate_angle(img)

# Entferne die Kontur-Erkennung und das Zeichnen
img_contour = img.copy()

# Interaktive Auswahl von zwei Punkten und Anzeige der Distanz in mm
print("Wähle zwei beliebige Punkte im Bild aus, um die Distanz zu messen (alle Angaben in mm).")
selected_points = []
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Umrechnung der Pixelkoordinaten in mm
        x_mm = x * scale
        y_mm = y * scale
        selected_points.append((x_mm, y_mm))
        # Markierung im Bild (optional, weiterhin an Pixelposition)
        cv2.circle(img_contour, (x, y), 7, (0, 0, 255), -1)
        cv2.imshow('Distanz messen', img_contour)

img_temp = img_contour.copy()
cv2.imshow('Distanz messen', img_temp)
cv2.setMouseCallback('Distanz messen', mouse_callback)
while len(selected_points) < 2:
    cv2.waitKey(1)
cv2.destroyAllWindows()

# Distanz berechnen und anzeigen (in mm)
p1 = np.array(selected_points[0])
p2 = np.array(selected_points[1])
dist_mm = np.linalg.norm(p1 - p2)
print(f"Distanz zwischen den gewählten Punkten: {dist_mm:.2f} mm")

# Linie und Text einzeichnen (weiterhin auf Pixelbasis, aber Text in mm)
pixel_p1 = (int(p1[0]/scale), int(p1[1]/scale))
pixel_p2 = (int(p2[0]/scale), int(p2[1]/scale))
cv2.line(img_contour, pixel_p1, pixel_p2, (0,0,255), 2)
cv2.putText(img_contour, f"{dist_mm:.1f} mm", ((pixel_p1[0]+pixel_p2[0])//2, (pixel_p1[1]+pixel_p2[1])//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
cv2.imshow('Gemessene Distanz', img_contour)
cv2.waitKey(3000)
cv2.destroyAllWindows()

# Interaktive Auswahl für Wandabstände
for wand_name in ["linken Wand", "rechten Wand"]:
    print(f"Wähle zwei Punkte für die Distanz zur {wand_name}.")
    wand_points = []
    def wand_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x_mm = x * scale
            y_mm = y * scale
            wand_points.append((x_mm, y_mm))
            cv2.circle(img_contour, (x, y), 7, (0, 255, 0), -1)
            cv2.imshow(f'Distanz zur {wand_name}', img_contour)
    img_temp = img_contour.copy()
    cv2.imshow(f'Distanz zur {wand_name}', img_temp)
    cv2.setMouseCallback(f'Distanz zur {wand_name}', wand_callback)
    while len(wand_points) < 2:
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    wp1 = np.array(wand_points[0])
    wp2 = np.array(wand_points[1])
    wand_dist_mm = np.linalg.norm(wp1 - wp2)
    print(f"Distanz zur {wand_name}: {wand_dist_mm:.2f} mm")
    pixel_wp1 = (int(wp1[0]/scale), int(wp1[1]/scale))
    pixel_wp2 = (int(wp2[0]/scale), int(wp2[1]/scale))
    cv2.line(img_contour, pixel_wp1, pixel_wp2, (0,255,0), 2)
    cv2.putText(img_contour, f"{wand_dist_mm:.1f} mm", ((pixel_wp1[0]+pixel_wp2[0])//2, (pixel_wp1[1]+pixel_wp2[1])//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow('Gemessene Wand-Distanzen', img_contour)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
