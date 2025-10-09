import os
import matplotlib.pyplot as plt
import numpy as np
from air_gap_measurement import main

DIRECTORY = rf"C:\Users\TBKAMMER\PycharmProjects\Spielkasten\FanVision_3D_IDS_Hackathon\data\air_gap\bildserie-3"
SAVE = True

def detect_gap_trend():
    image_files = []
    for filename in os.listdir(DIRECTORY):
        if filename.lower().endswith(".png"):
            image_files.append(filename)
    
    # Sort files to ensure consistent ordering
    image_files.sort()
    
    lengths = []
    index = []
    valid_files = []
    
    print(f"Processing {len(image_files)} images...")
    
    for i, file in enumerate(image_files):
        try:
            print(f"Processing image {i+1}/{len(image_files)}: {file}")
            _, _, shortest_info = main(os.path.join(DIRECTORY, file))
            
            if shortest_info and "length_mm" in shortest_info:
                print(f"  Result: {shortest_info}")
                lengths.append(shortest_info["length_mm"])
                index.append(i)
                valid_files.append(file)
            else:
                print(f"  Warning: No valid measurement for {file}")
                
        except Exception as e:
            print(f"  Error processing {file}: {e}")
    
    if not lengths:
        print("❌ No valid measurements found!")
        return
    
    # Calculate statistics
    max_length = max(lengths)
    min_length = min(lengths)
    avg_length = np.mean(lengths)
    max_index = lengths.index(max_length)
    max_file = valid_files[max_index]
    
    print(f"\n" + "="*50)
    print("AIR GAP TREND ANALYSIS RESULTS")
    print("="*50)
    print(f"Successfully processed: {len(lengths)} images")
    print(f"Maximum air gap: {max_length:.3f} mm (Image {max_index}: {max_file})")
    print(f"Minimum air gap: {min_length:.3f} mm")
    print(f"Average air gap: {avg_length:.3f} mm")
    print(f"Range: {max_length - min_length:.3f} mm")
    print("="*50)
    
    # Create the line chart
    plt.figure(figsize=(14, 8))
    
    # Main trend line
    plt.plot(index, lengths, 'b-o', linewidth=2, markersize=6, alpha=0.7, label='Air Gap Measurements')
    
    # Highlight the maximum point
    plt.plot(max_index, max_length, 'ro', markersize=12, alpha=0.8, label=f'Maximum: {max_length:.3f} mm')
    
    # Add horizontal lines for statistics
    plt.axhline(y=avg_length, color='green', linestyle='--', alpha=0.7, label=f'Average: {avg_length:.3f} mm')
    plt.axhline(y=max_length, color='red', linestyle=':', alpha=0.5, label=f'Max: {max_length:.3f} mm')
    plt.axhline(y=min_length, color='orange', linestyle=':', alpha=0.5, label=f'Min: {min_length:.3f} mm')
    
    # Annotations
    plt.annotate(f'MAX: {max_length:.3f} mm\n{max_file}', 
                xy=(max_index, max_length), 
                xytext=(max_index + len(index)*0.1, max_length + (max_length - min_length)*0.1),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Chart formatting
    plt.xlabel('Image Index', fontsize=12)
    plt.ylabel('Air Gap Length (mm)', fontsize=12)
    plt.title(f'Air Gap Trend Analysis\nDirectory: {os.path.basename(DIRECTORY)} ({len(lengths)} images)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    
    # Set x-axis to show all indices
    plt.xticks(index[::max(1, len(index)//20)])  # Show max 20 tick marks
    
    # Add some padding to y-axis
    y_range = max_length - min_length
    plt.ylim(min_length - y_range*0.1, max_length + y_range*0.2)
    
    # Show image names on x-axis if not too many images
    if len(valid_files) <= 15:
        plt.xticks(index, [f[:8] + '...' if len(f) > 8 else f for f in valid_files], rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save the plot
    if SAVE:
        output_path = os.path.join(DIRECTORY, "air_gap_trend_chart.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nChart saved to: {output_path}")
    
    plt.show()
    
    # Create a detailed data table
    print(f"\nDetailed Results:")
    print(f"{'Index':<6} {'Filename':<20} {'Length (mm)':<12}")
    print("-" * 40)
    for i, (idx, filename, length) in enumerate(zip(index, valid_files, lengths)):
        marker = " ← MAX" if length == max_length else ""
        print(f"{idx:<6} {filename[:18]:<20} {length:<12.3f}{marker}")
    
    return {
        'index': index,
        'lengths': lengths,
        'files': valid_files,
        'max_length': max_length,
        'max_index': max_index,
        'max_file': max_file,
        'statistics': {
            'min': min_length,
            'max': max_length,
            'avg': avg_length,
            'range': max_length - min_length
        }
    }

if __name__ == "__main__":
    results = detect_gap_trend()