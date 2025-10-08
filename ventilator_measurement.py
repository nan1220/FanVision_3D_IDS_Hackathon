import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "test_pic.png"  # Pfad zum Bild anpassen

img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"Fehler: Bild '{IMAGE_PATH}' konnte nicht geladen werden.")
    exit(1)

# Bild verkleinern
resize_factor = 0.1
img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)

# Vorverarbeitung
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (11, 11), 0) ## weich zeichnen 
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

cv2.imshow('thres', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                                      
# draw contours on the original image
image_copy = img.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                
# see the results
cv2.imshow('None approximation', image_copy)
cv2.waitKey(0)
#cv2.imwrite('contours_none_image1.jpg', image_copy)
cv2.destroyAllWindows()

# Referenzmaß: Durchmesser des eindeutigsten Kreises im thresh-Bild
REAL_DIAMETER_MM = 200  # Hier echten Wert eintragen!

# Kreisdetektion mit HoughCircles im thresh-Bild
circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                           param1=50, param2=30, minRadius=10, maxRadius=0)
if circles is not None:
    circles = np.uint16(np.around(circles))
    ref_circle = circles[0][0]
    ref_center = (int(ref_circle[0]), int(ref_circle[1]))
    ref_radius = int(ref_circle[2])
    pixel_diameter = ref_radius * 2
    scale = REAL_DIAMETER_MM / pixel_diameter
else:
    print("Kein Kreis eindeutig erkannt, Maßstab = 1")
    ref_center = (0, 0)
    ref_radius = 0
    scale = 1

img_annotated = img.copy()
if ref_radius > 0:
    cv2.circle(img_annotated, ref_center, ref_radius, (0, 0, 255), 2)
    cv2.putText(img_annotated, f"{REAL_DIAMETER_MM} mm", (ref_center[0], ref_center[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

min_length_px = 200  # Mindestumfang der Kontur in Pixel
for cnt in contours:
    if cv2.arcLength(cnt, True) < min_length_px:
        continue  # Kleine Konturen überspringen
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    # Nur Maße an Liniensegmenten der Konturen anzeigen
    for i in range(len(approx)):
        pt1 = tuple(approx[i][0])
        pt2 = tuple(approx[(i+1) % len(approx)][0])
        cv2.line(img_annotated, pt1, pt2, (0, 255, 0), 2)
        length_px = np.linalg.norm(np.array(pt1) - np.array(pt2))
        length_mm = length_px * scale
        mid_pt = ((pt1[0]+pt2[0])//2, (pt1[1]+pt2[1])//2)
        cv2.putText(img_annotated, f"{length_mm:.1f} mm", mid_pt,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Eindeutigsten Kreis rot markieren und Maß anschreiben
if ref_radius > 0:
    cv2.circle(img_annotated, ref_center, ref_radius, (0, 0, 255), 2)
    cv2.putText(img_annotated, f"{REAL_DIAMETER_MM} mm", (ref_center[0], ref_center[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

img_rgb = cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 10))
plt.imshow(img_rgb)
plt.title("Konturen als Linien/Kreise mit Maßangaben")
plt.axis('off')
plt.show()












