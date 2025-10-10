import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def preprocess_image_bw(gray_image, method='otsu'):
    """
    Preprocess the image to black and white using OpenCV's built-in methods.
    
    Args:
        gray_image: Grayscale image array
        method: Thresholding method ('otsu', 'adaptive_mean', 'adaptive_gaussian', 'triangle')
        
    Returns:
        binary_image: Binary (black and white) image
    """
    if method == 'otsu':
        # Otsu's automatic threshold selection
        threshold_value, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    elif method == 'adaptive_mean':
        # Adaptive threshold using mean of neighborhood
        binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        threshold_value = None  # Adaptive methods don't have a single threshold
        
    elif method == 'adaptive_gaussian':
        # Adaptive threshold using Gaussian-weighted sum of neighborhood
        binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        threshold_value = None
        
    elif method == 'triangle':
        # Triangle algorithm for automatic threshold selection
        threshold_value, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
            
    elif method == 'fixed':
        # Fixed threshold (fallback approach)
        threshold_value = 128
        _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'otsu', 'adaptive_mean', 'adaptive_gaussian', 'triangle', or 'fixed'")
    
    return binary_image

def crop_upper_part(image, crop_pixels=0, crop_percentage=0):
    """
    Remove the upper part of the image.
    
    Args:
        image: Input image (grayscale or color)
        crop_pixels: Number of pixels to remove from top (absolute value)
        crop_percentage: Percentage of height to remove from top (0-100)
        
    Returns:
        cropped_image: Image with upper part removed
    """
    height = image.shape[0]
    
    # Calculate crop amount
    if crop_percentage > 0:
        crop_amount = int(height * crop_percentage / 100)
    else:
        crop_amount = crop_pixels
    
    # Ensure we don't crop the entire image
    crop_amount = min(crop_amount, height - 1)
    
    if crop_amount > 0:
        return image[crop_amount:, :]
    else:
        return image

def measure_first_column_white_pixels(image):
    """
    Measure white pixels in the first column of a binary image.
    
    Args:
        image: Binary image (black and white)
        
    Returns:
        int: Number of white pixels in the first column
    """
    if len(image.shape) != 2:
        raise ValueError("Image must be grayscale/binary")
    
    # Get the first column (column index 0)
    first_column = image[:, 0]
    
    # Count white pixels (value 255)
    white_pixel_count = np.sum(first_column == 255)
    
    return white_pixel_count

def load_and_process_image(image_path, method='otsu', crop_pixels=0, crop_percentage=0):
    """
    Load an image and process it to black and white.
    
    Args:
        image_path: Path to the image file
        method: Thresholding method for cv2
        crop_pixels: Number of pixels to crop from top
        crop_percentage: Percentage of height to crop from top
        
    Returns:
        tuple: (original_gray, cropped_gray, processed_bw, white_pixel_count)
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Crop the upper part
    cropped_gray = crop_upper_part(gray, crop_pixels=crop_pixels, crop_percentage=crop_percentage)
    
    # Process to black and white using cv2 method
    processed_bw = preprocess_image_bw(cropped_gray, method=method)
    
    # Measure white pixels in first column
    white_pixel_count = measure_first_column_white_pixels(processed_bw)
    
    return gray, cropped_gray, processed_bw, white_pixel_count

def compare_two_images(image_path1, image_path2, method='otsu', crop_pixels=0, crop_percentage=0, scale_factor_mm_per_pixel=0.05, visualize=True):
    """
    Compare two images by processing them and measuring first column white pixels.
    
    Args:
        image_path1: Path to first image (slow)
        image_path2: Path to second image (fast)
        method: cv2 thresholding method ('otsu', 'adaptive_mean', 'adaptive_gaussian', 'triangle')
        crop_pixels: Number of pixels to crop from top (absolute)
        crop_percentage: Percentage of height to crop from top (0-100)
        scale_factor_mm_per_pixel: Scale factor for mm conversion
        visualize: Whether to show visualization plots
    """
    try:
        # Process both images
        gray1, cropped1, bw1, count1 = load_and_process_image(image_path1, method=method, crop_pixels=crop_pixels, crop_percentage=crop_percentage)
        gray2, cropped2, bw2, count2 = load_and_process_image(image_path2, method=method, crop_pixels=crop_pixels, crop_percentage=crop_percentage)
        
        # Convert pixel counts to mm
        count1_mm = count1 * scale_factor_mm_per_pixel
        count2_mm = count2 * scale_factor_mm_per_pixel
        
        # Calculate difference
        diff_pixels = abs(count1 - count2)
        diff_mm = diff_pixels * scale_factor_mm_per_pixel

        # Get image names for titles
        name1 = os.path.basename(image_path1)
        name2 = os.path.basename(image_path2)

        if visualize:
            # Create the comparison plot with 2 rows and 2 columns
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            crop_info = f"{crop_pixels}px" if crop_pixels > 0 else f"{crop_percentage}%"
            
            # Row 1: Unprocessed (cropped) images
            ax1.imshow(cropped1, cmap='gray')
            ax1.set_title(f'Slow - Unprocessed (cropped {crop_info} from top)\n{name1}', fontsize=12)
            ax1.axis('off')
            ax1.axvline(x=0, color='red', linewidth=3, alpha=0.7, label='First Column')
            ax1.legend()
            
            ax2.imshow(cropped2, cmap='gray')
            ax2.set_title(f'Fast - Unprocessed (cropped {crop_info} from top)\n{name2}', fontsize=12)
            ax2.axis('off')
            ax2.axvline(x=0, color='red', linewidth=3, alpha=0.7, label='First Column')
            ax2.legend()
            
            # Row 2: Processed images (black and white)
            ax3.imshow(bw1, cmap='gray')
            ax3.set_title(f'Slow - Processed (Method: {method})\nWhite pixels: {count1} ({count1_mm:.2f} mm)', fontsize=12)
            ax3.axis('off')
            
            # Highlight the first column and show white pixels
            ax3.axvline(x=0, color='red', linewidth=3, alpha=0.7)
            
            # Mark white pixels in the first column
            first_col_1 = bw1[:, 0]
            white_rows_1 = np.where(first_col_1 == 255)[0]
            if len(white_rows_1) > 0:
                ax3.scatter([0] * len(white_rows_1), white_rows_1, c='yellow', s=20, alpha=0.8, marker='s')
            
            ax4.imshow(bw2, cmap='gray')
            ax4.set_title(f'Fast - Processed (Method: {method})\nWhite pixels: {count2} ({count2_mm:.2f} mm)', fontsize=12)
            ax4.axis('off')
            
            # Highlight the first column and show white pixels
            ax4.axvline(x=0, color='red', linewidth=3, alpha=0.7)
            
            # Mark white pixels in the first column
            first_col_2 = bw2[:, 0]
            white_rows_2 = np.where(first_col_2 == 255)[0]
            if len(white_rows_2) > 0:
                ax4.scatter([0] * len(white_rows_2), white_rows_2, c='yellow', s=20, alpha=0.8, marker='s')
            
            plt.tight_layout()
            
            # Add overall title with comparison results
            plt.suptitle(f'First Column White Pixel Comparison\n'
                        f'Slow: {count1} px ({count1_mm:.2f} mm) | '
                        f'Fast: {count2} px ({count2_mm:.2f} mm) | '
                        f'Difference: {diff_pixels} px ({diff_mm:.2f} mm)', 
                        y=0.98, fontsize=14, fontweight='bold')
            
            plt.show()
        
        
        return {
            'image1': {
                'path': image_path1,
                'name': name1,
                'white_pixels': count1,
                'white_pixels_mm': count1_mm,
                'original_dimensions': gray1.shape,
                'cropped_dimensions': cropped1.shape
            },
            'image2': {
                'path': image_path2,
                'name': name2,
                'white_pixels': count2,
                'white_pixels_mm': count2_mm,
                'original_dimensions': gray2.shape,
                'cropped_dimensions': cropped2.shape
            },
            'comparison': {
                'difference_pixels': diff_pixels,
                'difference_mm': diff_mm,
                'relative_difference_percent': (diff_pixels/max(count1, count2)*100) if max(count1, count2) > 0 else 0
            },
            'settings': {
                'method': method,
                'crop_pixels': crop_pixels,
                'crop_percentage': crop_percentage,
                'scale_factor': scale_factor_mm_per_pixel
            }
        }
        
    except Exception as e:
        print(f"Error processing images: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualize_comparison_trend(white_pixels_slow, white_pixels_fast, scale_factor=0.05, save_path=None):
    """
    Create a comparison plot showing the trend of white pixels for both slow and fast series.
    
    Args:
        white_pixels_slow: List of white pixel counts for slow series
        white_pixels_fast: List of white pixel counts for fast series
        scale_factor: Scale factor for mm conversion
        save_path: Path to save the plot (optional)
    """
    # Create indices
    indices = list(range(len(white_pixels_slow)))
    
    # Convert to mm
    white_pixels_slow_mm = [px * scale_factor for px in white_pixels_slow]
    white_pixels_fast_mm = [px * scale_factor for px in white_pixels_fast]
    
    # Calculate statistics
    avg_slow = np.mean(white_pixels_slow_mm)
    avg_fast = np.mean(white_pixels_fast_mm)
    std_slow = np.std(white_pixels_slow_mm)
    std_fast = np.std(white_pixels_fast_mm)
    
    # Create the plot with single subplot
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Plot: White pixels comparison trend (mm)
    ax.plot(indices, white_pixels_slow_mm, 'b-o', linewidth=2, markersize=6, alpha=0.7, label='Slow')
    ax.plot(indices, white_pixels_fast_mm, 'r-s', linewidth=2, markersize=6, alpha=0.7, label='Fast')
    
    # Add average lines
    ax.axhline(y=avg_slow, color='blue', linestyle='--', alpha=0.5, label=f'Avg Slow: {avg_slow:.2f} mm')
    ax.axhline(y=avg_fast, color='red', linestyle='--', alpha=0.5, label=f'Avg Fast: {avg_fast:.2f} mm')
    
    ax.set_xlabel('Image Index', fontsize=12)
    ax.set_ylabel('White Pixels in First Column (mm)', fontsize=12)
    ax.set_title('Air Gap Comparison Trend: Slow vs Fast', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_xticks(indices)
    
    # Add statistics as text box
    stats_text = f'Slow: n={len(white_pixels_slow)}, min={min(white_pixels_slow_mm):.2f} mm, max={max(white_pixels_slow_mm):.2f} mm, std={std_slow:.2f} mm\n'
    stats_text += f'Fast: n={len(white_pixels_fast)}, min={min(white_pixels_fast_mm):.2f} mm, max={max(white_pixels_fast_mm):.2f} mm, std={std_fast:.2f} mm\n'
    stats_text += f'Avg Difference: {abs(avg_slow - avg_fast):.2f} mm ({abs(avg_slow - avg_fast)/avg_slow*100:.1f}%)'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    # Save the plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to: {save_path}")
    
    plt.show()
    
    # Print detailed statistics
    print("\n" + "="*60)
    print("AIR GAP COMPARISON STATISTICS")
    print("="*60)
    print(f"Number of images processed: {len(indices)}")
    print(f"\nSlow Series:")
    print(f"  Average: {avg_slow:.3f} mm ({np.mean(white_pixels_slow):.1f} px)")
    print(f"  Min: {min(white_pixels_slow_mm):.3f} mm ({min(white_pixels_slow)} px)")
    print(f"  Max: {max(white_pixels_slow_mm):.3f} mm ({max(white_pixels_slow)} px)")
    print(f"  Std Dev: {std_slow:.3f} mm")
    print(f"\nFast Series:")
    print(f"  Average: {avg_fast:.3f} mm ({np.mean(white_pixels_fast):.1f} px)")
    print(f"  Min: {min(white_pixels_fast_mm):.3f} mm ({min(white_pixels_fast)} px)")
    print(f"  Max: {max(white_pixels_fast_mm):.3f} mm ({max(white_pixels_fast)} px)")
    print(f"  Std Dev: {std_fast:.3f} mm")
    print(f"\nDifferences:")
    differences_mm = [abs(s - f) * scale_factor for s, f in zip(white_pixels_slow, white_pixels_fast)]
    print(f"  Average difference: {np.mean(differences_mm):.3f} mm")
    print(f"  Max difference: {max(differences_mm):.3f} mm")
    print(f"  Min difference: {min(differences_mm):.3f} mm")
    print(f"  Difference in averages: {abs(avg_slow - avg_fast):.3f} mm ({abs(avg_slow - avg_fast)/avg_slow*100:.1f}%)")
    print("="*60)
    
    # Print detailed table
    print(f"\nDetailed Comparison:")
    print(f"{'Index':<6} {'Slow (mm)':<12} {'Fast (mm)':<12} {'Diff (mm)':<12} {'Diff %':<10}")
    print("-" * 60)
    for i, (slow, fast) in enumerate(zip(white_pixels_slow, white_pixels_fast)):
        slow_mm = slow * scale_factor
        fast_mm = fast * scale_factor
        diff_mm = abs(slow_mm - fast_mm)
        diff_pct = (diff_mm / slow_mm * 100) if slow_mm > 0 else 0
        print(f"{i:<6} {slow_mm:<12.3f} {fast_mm:<12.3f} {diff_mm:<12.3f} {diff_pct:<10.1f}")

def main():
    """
    Main function to compare two images.
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Processing parameters
    method = 'otsu'  # Options: 'otsu', 'adaptive_mean', 'adaptive_gaussian', 'triangle', 'fixed'
    crop_pixels = 0  # Number of pixels to remove from top (use this OR crop_percentage)
    crop_percentage = 20  # Percentage of height to remove from top (0-100)
    scale_factor = 0.05  # mm per pixel - adjust based on your image scale

    visualize = False  # Set to True to see individual image comparisons

    if crop_percentage > 0:
        print(f"Cropping: {crop_percentage}% from top")
    elif crop_pixels > 0:
        print(f"Cropping: {crop_pixels} pixels from top")
    
    white_pixels_slow = []
    white_pixels_fast = []
    
    print("Processing image pairs...")
    for i in range(10):
        image_nr_slow = f"{i+21}" if i+1 < 10 else f"30"
        image_nr_fast = f"{i+11}" if i+1 < 10 else f"20"
        image1_path = os.path.join(script_dir, "data", "air_gap", "bildserie-trigger", "langsam", f"spalt_2{image_nr_slow}.png")
        image2_path = os.path.join(script_dir, "data", "air_gap", "bildserie-trigger", "schnell", f"spalt_2{image_nr_fast}.png")
        
        print(f"\nPair {i+1}/10:")
        print(f"  Slow: {os.path.basename(image1_path)}")
        print(f"  Fast: {os.path.basename(image2_path)}")
        
        # Compare the images
        results = compare_two_images(
            image1_path, 
            image2_path,
            method=method,
            crop_pixels=crop_pixels,
            crop_percentage=crop_percentage,
            scale_factor_mm_per_pixel=scale_factor,
            visualize=visualize
        )
        
        if results:
            white_pixels_slow.append(results["image1"]["white_pixels"])
            white_pixels_fast.append(results["image2"]["white_pixels"])
            print(f"  Slow: {results['image1']['white_pixels']} px ({results['image1']['white_pixels_mm']:.2f} mm)")
            print(f"  Fast: {results['image2']['white_pixels']} px ({results['image2']['white_pixels_mm']:.2f} mm)")
            print(f"  Diff: {results['comparison']['difference_pixels']} px ({results['comparison']['difference_mm']:.2f} mm)")
    
    # Visualize the comparison trend
    if white_pixels_slow and white_pixels_fast:
        save_path = os.path.join(script_dir, "data", "air_gap", "bildserie-trigger", "comparison_trend.png")
        visualize_comparison_trend(white_pixels_slow, white_pixels_fast, scale_factor=scale_factor, save_path=save_path)
    
    return white_pixels_slow, white_pixels_fast

if __name__ == "__main__":
    white_pixels_slow, white_pixels_fast = main()