import numpy as np
import cv2

def rewarpping_image(image, points=None, output_site_px=500):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    # --- Find red pixels (assumes you marked exactly 4 corners in red) --- 
    mask = (image[:, :, 0] == 0) & (image[:, :, 1] == 0) & (image[:, :, 2] == 255) 
    if not points:
        points = np.argwhere(mask) 
        # gives (y, x) 
        if len(points) != 4: 
            raise ValueError(f"Expected 4 red points, found {len(points)}") 
        # Sort points roughly in top-left, top-right, bottom-right, bottom-left order 
        # We'll sort by y first (top vs bottom), then x 
        points = sorted(points, key=lambda p: (p[0], p[1])) 
    top = sorted(points[:2], key=lambda p: p[1]) 
    # top-left, top-right 
    bottom = sorted(points[2:], key=lambda p: p[1]) 
    # bottom-left, bottom-right 
    pts_src = np.array([top[0][::-1], top[1][::-1], bottom[1][::-1], bottom[0][::-1]], dtype=np.float32) 
    # [::-1] converts (y, x) → (x, y) for OpenCV 
    # --- Define destination points (perfect square, 10x10 cm scaled arbitrarily to pixels) ---     
    pts_dst = np.array([ [0, 0], [output_site_px, 0], [output_site_px, output_site_px], [0, output_site_px] ], dtype=np.float32) 
    # --- Compute homography and warp image ---
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