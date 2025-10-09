import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(gray_image, white_threshold=200, black_threshold=50):
    """
    Preprocess the image to make grey areas black, keeping only white and black regions.
    
    Args:
        gray_image: Grayscale image array
        white_threshold: Pixels above this value become white (255)
        black_threshold: Pixels below this value become black (0)
        
    Returns:
        preprocessed_image: Binary image with only black and white pixels
    """
    preprocessed = gray_image.copy()
    
    # Make truly white pixels white (255)
    preprocessed[gray_image >= white_threshold] = 255
    
    # Make dark pixels black (0)
    preprocessed[gray_image <= black_threshold] = 0
    
    # Make grey pixels (between thresholds) black
    grey_mask = (gray_image > black_threshold) & (gray_image < white_threshold)
    preprocessed[grey_mask] = 0
    
    return preprocessed

def measure_white_pixels_per_row(image_path, white_threshold=200, black_threshold=50):
    """
    Measure the length of white pixels in each row of an image.
    
    Args:
        image_path (str): Path to the image file
        white_threshold (int): Threshold value to consider a pixel as white (0-255)
        black_threshold (int): Threshold value to consider a pixel as black (0-255)
    
    Returns:
        tuple: (white_pixel_lengths, image_height, image_width, original_gray, preprocessed_gray)
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Preprocess image to remove grey areas
    preprocessed_gray = preprocess_image(gray, white_threshold, black_threshold)
    
    # Get image dimensions
    height, width = preprocessed_gray.shape
    
    # Create binary mask for white pixels (using preprocessed image)
    white_mask = preprocessed_gray >= white_threshold
    
    # Measure white pixel length for each row
    white_pixel_lengths = []
    
    for row in range(height):
        # Get the current row
        row_data = white_mask[row, :]
        
        # Count consecutive white pixels
        # Option 1: Total white pixels in the row
        total_white = np.sum(row_data)
        
        # Option 2: Longest consecutive white pixel sequence
        # Find consecutive sequences
        consecutive_lengths = []
        current_length = 0
        
        for pixel in row_data:
            if pixel:  # White pixel
                current_length += 1
            else:
                if current_length > 0:
                    consecutive_lengths.append(current_length)
                current_length = 0
        
        # Don't forget the last sequence if it ends with white pixels
        if current_length > 0:
            consecutive_lengths.append(current_length)
        
        # Store both total white pixels and longest consecutive sequence
        max_consecutive = max(consecutive_lengths) if consecutive_lengths else 0
        white_pixel_lengths.append({
            'row': row,
            'total_white_pixels': total_white,
            'max_consecutive_white': max_consecutive,
            'all_consecutive_sequences': consecutive_lengths
        })
    
    return white_pixel_lengths, height, width, gray, preprocessed_gray

def find_longest_white_sequence(white_pixel_lengths):
    """
    Find the longest white sequence across all rows and its position.
    
    Returns:
        dict: Information about the longest sequence including row, start, end, length
    """
    longest_sequence = {
        'row': -1,
        'start_col': -1,
        'end_col': -1,
        'length': 0
    }
    
    for data in white_pixel_lengths:
        if data['max_consecutive_white'] > longest_sequence['length']:
            longest_sequence['length'] = data['max_consecutive_white']
            longest_sequence['row'] = data['row']
            
            # Find the position of the longest sequence in this row
            sequences = data['all_consecutive_sequences']
            if sequences:
                max_length = max(sequences)
                # We need to find where this sequence starts in the original row
                # This requires re-analyzing the row to get positions
    
    return longest_sequence

def get_sequence_positions(preprocessed_image, row_index, white_threshold=240):
    """
    Get the start and end positions of white sequences in a specific row.
    """
    white_mask = preprocessed_image >= white_threshold
    row_data = white_mask[row_index, :]
    
    sequences = []
    start_pos = None
    
    for col, pixel in enumerate(row_data):
        if pixel:  # White pixel
            if start_pos is None:
                start_pos = col
        else:  # Non-white pixel
            if start_pos is not None:
                sequences.append({
                    'start': start_pos,
                    'end': col - 1,
                    'length': col - start_pos
                })
                start_pos = None
    
    # Handle case where row ends with white pixels
    if start_pos is not None:
        sequences.append({
            'start': start_pos,
            'end': len(row_data) - 1,
            'length': len(row_data) - start_pos
        })
    
    return sequences

def visualize_measurements(white_pixel_lengths, image_height, preprocessed_image, original_image):
    """
    Create visualizations of the white pixel measurements with both original and preprocessed images.
    """
    rows = [data['row'] for data in white_pixel_lengths]
    total_whites = [data['total_white_pixels'] for data in white_pixel_lengths]
    max_consecutives = [data['max_consecutive_white'] for data in white_pixel_lengths]
    
    # Find the shortest white sequence across all rows (excluding rows with no white pixels)
    shortest_overall = float('inf')
    shortest_row = -1
    for data in white_pixel_lengths:
        if data['max_consecutive_white'] > 0 and data['max_consecutive_white'] < shortest_overall:
            shortest_overall = data['max_consecutive_white']
            shortest_row = data['row']
    
    # Handle case where no white pixels found
    if shortest_overall == float('inf'):
        shortest_overall = 0
        shortest_row = -1
    
    # Get the exact position of the shortest sequence
    shortest_sequence_info = None
    if shortest_row >= 0:
        sequences = get_sequence_positions(preprocessed_image, shortest_row)
        for seq in sequences:
            if seq['length'] == shortest_overall:
                shortest_sequence_info = {
                    'row': shortest_row,
                    'start': seq['start'],
                    'end': seq['end'],
                    'length': seq['length']
                }
                break
    
    # Create subplots - 4 plots in one figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Original image
    ax1.imshow(original_image, cmap='gray')
    ax1.set_title('Original Image (Grayscale)')
    ax1.set_xlabel('Width (pixels)')
    ax1.set_ylabel('Height (pixels)')
    
    # Plot 2: Preprocessed image with shortest sequence marked
    ax2.imshow(preprocessed_image, cmap='gray')
    ax2.set_title('Preprocessed Image (Grey→Black)')
    ax2.set_xlabel('Width (pixels)')
    ax2.set_ylabel('Height (pixels)')
    
    # Mark the shortest white sequence on the preprocessed image
    if shortest_sequence_info:
        row = shortest_sequence_info['row']
        start = shortest_sequence_info['start']
        end = shortest_sequence_info['end']
        
        # Draw a red rectangle around the shortest sequence
        rect = plt.Rectangle((start, row-1), end-start+1, 3, 
                           fill=False, edgecolor='red', linewidth=2)
        ax2.add_patch(rect)
        
        # Add text annotation
        ax2.annotate(f'Shortest: {shortest_sequence_info["length"]} pixels', 
                    xy=(start, row), xytext=(start, row-20),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    color='red', fontweight='bold')
    
    # Plot 3: Total white pixels per row
    ax3.plot(total_whites, rows, 'b-', linewidth=2)
    ax3.set_xlabel('Total White Pixels')
    ax3.set_ylabel('Row (Height)')
    ax3.set_title('Total White Pixels per Row')
    ax3.invert_yaxis()  # Invert y-axis to match image orientation
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([image_height, 0])  # Match image height
    
    # Mark the row with shortest sequence
    if shortest_row >= 0:
        ax3.axhline(y=shortest_row, color='red', linestyle='--', alpha=0.7)
        ax3.annotate(f'Shortest sequence row', xy=(total_whites[shortest_row], shortest_row), 
                    xytext=(total_whites[shortest_row] + max(total_whites)*0.1, shortest_row),
                    arrowprops=dict(arrowstyle='->', color='red'), color='red')
    
    # Plot 4: Maximum consecutive white pixels per row
    ax4.plot(max_consecutives, rows, 'r-', linewidth=2)
    ax4.set_xlabel('Max Consecutive White Pixels')
    ax4.set_ylabel('Row (Height)')
    ax4.set_title('Consecutive White Pixel Sequence per Row')
    ax4.invert_yaxis()  # Invert y-axis to match image orientation
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([image_height, 0])  # Match image height
    
    # Highlight the shortest sequence point
    if shortest_row >= 0:
        ax4.plot(shortest_overall, shortest_row, 'ro', markersize=10, markerfacecolor='red')
        ax4.annotate(f'Min: {shortest_overall} pixels', 
                    xy=(shortest_overall, shortest_row), 
                    xytext=(shortest_overall + max(max_consecutives)*0.1, shortest_row),
                    arrowprops=dict(arrowstyle='->', color='red'), 
                    color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.suptitle(f'White Pixel Analysis - Shortest Sequence Highlighted', y=0.98, fontsize=16)
    plt.show()
    
    # Print shortest sequence information
    if shortest_sequence_info:
        print(f"\nShortest white sequence found:")
        print(f"Row: {shortest_sequence_info['row']}")
        print(f"Start column: {shortest_sequence_info['start']}")
        print(f"End column: {shortest_sequence_info['end']}")
        print(f"Length: {shortest_sequence_info['length']} pixels")

def detect_outliers(white_pixel_lengths, outlier_method='row_change', threshold_factor=0.5, min_pixel_threshold=10, outlier_expansion=5):
    """
    Detect outlier rows based on changes from the previous row and minimum pixel length.
    Also expands outlier detection to include nearby rows.
    
    Args:
        white_pixel_lengths: List of measurement data for each row
        outlier_method: 'row_change', 'row_change_relative', or 'row_change_adaptive'
        threshold_factor: Factor for outlier detection sensitivity
        min_pixel_threshold: Minimum white pixel sequence length (sequences below this are outliers)
        outlier_expansion: Number of rows around each outlier to also mark as outliers
    
    Returns:
        dict: Contains outlier information and filtered data
    """
    max_consecutives = [data['max_consecutive_white'] for data in white_pixel_lengths]
    total_whites = [data['total_white_pixels'] for data in white_pixel_lengths]
    
    initial_outlier_rows = []
    
    # First, classify rows with sequences shorter than min_pixel_threshold as outliers
    for i, data in enumerate(white_pixel_lengths):
        if 0 < data['max_consecutive_white'] < min_pixel_threshold:
            initial_outlier_rows.append(i)
    
    # Method 1: Absolute row-to-row change detection
    if outlier_method == 'row_change':
        for i in range(1, len(max_consecutives)):
            current_value = max_consecutives[i]
            previous_value = max_consecutives[i-1]
            
            # Skip if already classified as outlier due to minimum pixel threshold
            if i in initial_outlier_rows:
                continue
            
            # Skip if both values are zero (no white pixels)
            if current_value == 0 and previous_value == 0:
                continue
            
            # Calculate absolute change
            change = abs(current_value - previous_value)
            
            # Calculate threshold based on the average of nearby values
            window_start = max(0, i-3)
            window_end = min(len(max_consecutives), i+4)
            nearby_values = [x for x in max_consecutives[window_start:window_end] if x >= min_pixel_threshold]
            
            if len(nearby_values) > 0:
                avg_nearby = np.mean(nearby_values)
                threshold = threshold_factor * avg_nearby
                
                if change > threshold:
                    initial_outlier_rows.append(i)
    
    # Method 2: Relative row-to-row change detection
    elif outlier_method == 'row_change_relative':
        for i in range(1, len(max_consecutives)):
            current_value = max_consecutives[i]
            previous_value = max_consecutives[i-1]
            
            # Skip if already classified as outlier due to minimum pixel threshold
            if i in initial_outlier_rows:
                continue
            
            # Skip if both values are zero
            if current_value == 0 and previous_value == 0:
                continue
                
            # Skip if previous value is zero but current isn't (start of white area)
            if previous_value == 0 and current_value > 0:
                continue
                
            # Skip if current value is zero but previous isn't (end of white area)
            if current_value == 0 and previous_value > 0:
                continue
            
            # Calculate relative change (only for values above min threshold)
            if previous_value >= min_pixel_threshold:
                relative_change = abs(current_value - previous_value) / previous_value
                
                if relative_change > threshold_factor:
                    initial_outlier_rows.append(i)
    
    # Method 3: Adaptive row-to-row change detection with smoothing
    elif outlier_method == 'row_change_adaptive':
        # Calculate moving average for smoother comparison (excluding short sequences)
        window_size = 5
        smoothed_values = []
        
        for i in range(len(max_consecutives)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(max_consecutives), i + window_size // 2 + 1)
            window_values = [x for x in max_consecutives[start_idx:end_idx] if x >= min_pixel_threshold]
            
            if window_values:
                smoothed_values.append(np.mean(window_values))
            else:
                smoothed_values.append(0)
        
        # Detect outliers based on deviation from smoothed trend
        for i in range(1, len(max_consecutives)):
            # Skip if already classified as outlier due to minimum pixel threshold
            if i in initial_outlier_rows:
                continue
                
            current_value = max_consecutives[i]
            smoothed_value = smoothed_values[i]
            
            if current_value >= min_pixel_threshold and smoothed_value > 0:
                deviation = abs(current_value - smoothed_value) / smoothed_value
                
                if deviation > threshold_factor:
                    initial_outlier_rows.append(i)
    
    # Method 4: Gradient-based detection
    elif outlier_method == 'gradient':
        # Calculate gradient (rate of change) between consecutive rows
        gradients = []
        for i in range(1, len(max_consecutives)):
            current_value = max_consecutives[i]
            previous_value = max_consecutives[i-1]
            gradient = current_value - previous_value
            gradients.append(gradient)
        
        # Calculate threshold based on gradient statistics (excluding short sequences)
        valid_gradients = []
        for i, gradient in enumerate(gradients):
            if max_consecutives[i] >= min_pixel_threshold and max_consecutives[i+1] >= min_pixel_threshold:
                valid_gradients.append(abs(gradient))
        
        if len(valid_gradients) > 0:
            gradient_threshold = threshold_factor * np.std(valid_gradients)
            
            for i, gradient in enumerate(gradients):
                # Skip if already classified as outlier due to minimum pixel threshold
                if (i + 1) in initial_outlier_rows:
                    continue
                    
                if abs(gradient) > gradient_threshold:
                    initial_outlier_rows.append(i + 1)  # +1 because gradients start from index 1
    
    # Remove duplicates from initial outliers
    initial_outlier_rows = sorted(list(set(initial_outlier_rows)))
    
    # EXPAND OUTLIERS: Include nearby rows around each outlier
    expanded_outlier_rows = set(initial_outlier_rows)
    
    for outlier_row in initial_outlier_rows:
        # Add rows before and after the outlier
        for offset in range(-outlier_expansion, outlier_expansion + 1):
            neighbor_row = outlier_row + offset
            # Make sure the neighbor row is within valid range
            if 0 <= neighbor_row < len(white_pixel_lengths):
                expanded_outlier_rows.add(neighbor_row)
    
    # Convert back to sorted list
    final_outlier_rows = sorted(list(expanded_outlier_rows))
    
    # Create filtered data (excluding all outliers)
    filtered_data = [data for i, data in enumerate(white_pixel_lengths) if i not in final_outlier_rows]
    
    return {
        'outlier_rows': final_outlier_rows,
        'initial_outlier_rows': initial_outlier_rows,  # Keep track of original outliers
        'expanded_outlier_rows': list(expanded_outlier_rows - set(initial_outlier_rows)),  # Only the expanded ones
        'filtered_data': filtered_data,
        'method': outlier_method,
        'threshold_factor': threshold_factor,
        'min_pixel_threshold': min_pixel_threshold,
        'outlier_expansion': outlier_expansion,
        'max_consecutives': max_consecutives
    }

def find_shortest_sequence_filtered(white_pixel_lengths, outlier_info):
    """
    Find the shortest white sequence excluding outlier rows.
    """
    filtered_data = outlier_info['filtered_data']
    
    shortest_overall = float('inf')
    shortest_row = -1
    
    for data in filtered_data:
        if data['max_consecutive_white'] > 0 and data['max_consecutive_white'] < shortest_overall:
            shortest_overall = data['max_consecutive_white']
            shortest_row = data['row']
    
    if shortest_overall == float('inf'):
        return None
    
    return {
        'row': shortest_row,
        'length': shortest_overall
    }

def visualize_measurements_with_outliers(white_pixel_lengths, image_height, preprocessed_image, original_image, outlier_method='row_change', threshold_factor=0.5, min_pixel_threshold=10, outlier_expansion=5):
    """
    Create visualizations highlighting outliers and showing filtered shortest sequence.
    """
    rows = [data['row'] for data in white_pixel_lengths]
    total_whites = [data['total_white_pixels'] for data in white_pixel_lengths]
    max_consecutives = [data['max_consecutive_white'] for data in white_pixel_lengths]
    
    # Detect outliers (including minimum pixel threshold and expansion)
    outlier_info = detect_outliers(white_pixel_lengths, outlier_method=outlier_method, threshold_factor=threshold_factor, min_pixel_threshold=min_pixel_threshold, outlier_expansion=outlier_expansion)
    
    outlier_rows = outlier_info['outlier_rows']
    initial_outlier_rows = outlier_info['initial_outlier_rows']
    expanded_outlier_rows = outlier_info['expanded_outlier_rows']
    
    # Separate initial outliers by type for better visualization
    min_pixel_outliers = []
    change_outliers = []
    
    for outlier_row in initial_outlier_rows:
        if 0 < max_consecutives[outlier_row] < min_pixel_threshold:
            min_pixel_outliers.append(outlier_row)
        else:
            change_outliers.append(outlier_row)
    
    # Find shortest sequence excluding all outliers (including expanded)
    shortest_filtered = find_shortest_sequence_filtered(white_pixel_lengths, outlier_info)
    
    # Get sequence position for the filtered shortest
    shortest_sequence_info = None
    if shortest_filtered:
        sequences = get_sequence_positions(preprocessed_image, shortest_filtered['row'])
        for seq in sequences:
            if seq['length'] == shortest_filtered['length']:
                shortest_sequence_info = {
                    'row': shortest_filtered['row'],
                    'start': seq['start'],
                    'end': seq['end'],
                    'length': seq['length']
                }
                break
    
    # Create subplots with additional change analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Original image
    ax1.imshow(original_image, cmap='gray')
    ax1.set_title('Original Image (Grayscale)')
    ax1.set_xlabel('Width (pixels)')
    ax1.set_ylabel('Height (pixels)')
    
    # Plot 2: Preprocessed image with shortest sequence and outliers marked
    ax2.imshow(preprocessed_image, cmap='gray')
    ax2.set_title(f'Preprocessed Image with Expanded Outliers (±{outlier_expansion} rows)')
    ax2.set_xlabel('Width (pixels)')
    ax2.set_ylabel('Height (pixels)')
    
    # Mark different types of outlier rows on the image
    for outlier_row in min_pixel_outliers:
        ax2.axhline(y=outlier_row, color='yellow', alpha=0.6, linewidth=2, label=f'Min pixels < {min_pixel_threshold}' if outlier_row == min_pixel_outliers[0] else "")
    
    for outlier_row in change_outliers:
        ax2.axhline(y=outlier_row, color='orange', alpha=0.7, linewidth=2, label='Row-change outlier' if outlier_row == change_outliers[0] else "")
    
    # Mark expanded outlier areas
    for outlier_row in expanded_outlier_rows:
        ax2.axhline(y=outlier_row, color='lightcoral', alpha=0.3, linewidth=1, label=f'Expanded outliers (±{outlier_expansion})' if outlier_row == expanded_outlier_rows[0] else "")
    
    # Mark the shortest sequence (excluding all outliers)
    if shortest_sequence_info:
        row = shortest_sequence_info['row']
        start = shortest_sequence_info['start']
        end = shortest_sequence_info['end']
        
        rect = plt.Rectangle((start, row-1), end-start+1, 3, 
                           fill=False, edgecolor='red', linewidth=3)
        ax2.add_patch(rect)
        
        ax2.annotate(f'Shortest (filtered): {shortest_sequence_info["length"]} pixels', 
                    xy=(start, row), xytext=(start, row-30),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    color='red', fontweight='bold')
    
    if outlier_rows:
        ax2.legend()
    
    # Plot 3: Max consecutive white pixels with all outliers highlighted
    ax3.plot(max_consecutives, rows, 'b-', linewidth=1, alpha=0.7, label='All data')
    
    # Highlight different types of outliers
    if min_pixel_outliers:
        min_pixel_consecutives = [max_consecutives[i] for i in min_pixel_outliers]
        min_pixel_row_nums = [rows[i] for i in min_pixel_outliers]
        ax3.scatter(min_pixel_consecutives, min_pixel_row_nums, color='yellow', s=50, alpha=0.8, label=f'Min pixels < {min_pixel_threshold}', zorder=6)
    
    if change_outliers:
        change_consecutives = [max_consecutives[i] for i in change_outliers]
        change_row_nums = [rows[i] for i in change_outliers]
        ax3.scatter(change_consecutives, change_row_nums, color='orange', s=60, alpha=0.8, label='Row-change outliers', zorder=6)
    
    # Highlight expanded outliers
    if expanded_outlier_rows:
        expanded_consecutives = [max_consecutives[i] for i in expanded_outlier_rows]
        expanded_row_nums = [rows[i] for i in expanded_outlier_rows]
        ax3.scatter(expanded_consecutives, expanded_row_nums, color='lightcoral', s=20, alpha=0.6, label=f'Expanded outliers (±{outlier_expansion})', zorder=5)
    
    # Add threshold line
    ax3.axvline(x=min_pixel_threshold, color='red', linestyle=':', alpha=0.7, label=f'Min threshold: {min_pixel_threshold}')
    
    ax3.set_xlabel('Max Consecutive White Pixels')
    ax3.set_ylabel('Row (Height)')
    ax3.set_title(f'Consecutive White Pixels (Expanded Outlier Detection)')
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([image_height, 0])
    ax3.legend()
    
    # Plot 4: Row-to-row changes
    changes = []
    change_rows = []
    for i in range(1, len(max_consecutives)):
        change = max_consecutives[i] - max_consecutives[i-1]
        changes.append(change)
        change_rows.append(rows[i])
    
    ax4.plot(changes, change_rows, 'g-', linewidth=1, alpha=0.7, label='Row-to-row change')
    ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Highlight outlier changes
    initial_change_outlier_changes = [changes[i-1] for i in initial_outlier_rows if i-1 < len(changes) and i > 0]
    initial_change_outlier_rows = [rows[i] for i in initial_outlier_rows if i-1 < len(changes) and i > 0]
    ax4.scatter(initial_change_outlier_changes, initial_change_outlier_rows, color='orange', s=50, alpha=0.8, label='Initial outlier changes', zorder=5)
    
    ax4.set_xlabel('Change in White Pixels (current - previous)')
    ax4.set_ylabel('Row (Height)')
    ax4.set_title(f'Row-to-Row Changes (Expansion: ±{outlier_expansion} rows)')
    ax4.invert_yaxis()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([image_height, 0])
    ax4.legend()
    
    plt.tight_layout()
    plt.suptitle(f'Expanded Outlier Detection ({len(outlier_rows)} total: {len(initial_outlier_rows)} initial + {len(expanded_outlier_rows)} expanded)', y=0.98, fontsize=16)
    plt.show()
    
    # Print detailed results
    print(f"\nExpanded Outlier Detection Results:")
    print(f"Method: {outlier_method}")
    print(f"Threshold factor: {threshold_factor}")
    print(f"Minimum pixel threshold: {min_pixel_threshold}")
    print(f"Outlier expansion: ±{outlier_expansion} rows")
    print(f"Initial outliers detected: {len(initial_outlier_rows)}")
    print(f"Expanded outliers added: {len(expanded_outlier_rows)}")
    print(f"Total outliers: {len(outlier_rows)}")
    print(f"Min-pixel outliers: {len(min_pixel_outliers)} rows")
    print(f"Row-change outliers: {len(change_outliers)} rows")
    
    # Show details of different types of outliers
    if min_pixel_outliers:
        print(f"\nMin-pixel outliers (< {min_pixel_threshold} pixels):")
        for outlier_row in min_pixel_outliers[:10]:  # Show first 10
            pixels = max_consecutives[outlier_row]
            print(f"Row {outlier_row}: {pixels} pixels")
    
    if change_outliers:
        print(f"\nRow-change outliers:")
        for outlier_row in change_outliers[:10]:  # Show first 10
            if outlier_row > 0 and outlier_row < len(max_consecutives):
                prev_val = max_consecutives[outlier_row-1]
                curr_val = max_consecutives[outlier_row]
                change = curr_val - prev_val
                print(f"Row {outlier_row}: {prev_val} → {curr_val} (change: {change:+d})")
    
    if shortest_sequence_info:
        print(f"\nShortest white sequence (all outliers filtered):")
        print(f"Row: {shortest_sequence_info['row']}")
        print(f"Start column: {shortest_sequence_info['start']}")
        print(f"End column: {shortest_sequence_info['end']}")
        print(f"Length: {shortest_sequence_info['length']} pixels")
    
    return outlier_info, shortest_sequence_info

def main():
    # Example usage
    image_path = "image_54.png"  # Change this to your image path
    
    try:
        # Measure white pixels
        measurements, height, width, original_gray, preprocessed_gray = measure_white_pixels_per_row(image_path)
        
        print(f"Image dimensions: {width} x {height}")
        print(f"Analyzed {len(measurements)} rows")
        
        # Display statistics
        total_whites = [data['total_white_pixels'] for data in measurements]
        max_consecutives = [data['max_consecutive_white'] for data in measurements]
        non_zero_consecutives = [x for x in max_consecutives if x > 0]
        
        print(f"\nStatistics (before outlier filtering):")
        print(f"Rows with white pixels: {len(non_zero_consecutives)}")
        print(f"Shortest consecutive white pixels: {min(non_zero_consecutives) if non_zero_consecutives else 0}")
        print(f"Longest consecutive white pixels: {max(max_consecutives)}")
        
        # Visualize with expanded outlier detection (row-change + minimum pixels + expansion)
        outlier_info, shortest_info = visualize_measurements_with_outliers(
            measurements, height, preprocessed_gray, original_gray, 
            outlier_method='row_change', threshold_factor=0.3, min_pixel_threshold=10, outlier_expansion=5)
        
        return measurements, outlier_info
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None

if __name__ == "__main__":
    main()