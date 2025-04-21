from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def separate_palimpsest(image_path, output_dir='results', manual_channel_selection=None, manual_scale_factor=None):
    """
    A simplified approach to separate underwriting in palimpsest images using RGB channel analysis
    
    Parameters:
    -----------
    image_path : str
        Path to the input palimpsest image
    output_dir : str
        Directory to save output images
    manual_channel_selection : tuple or None
        Manually specify which channels to use (e.g., ('R', 'G') or ('B', 'R'))
        If None, will automatically select based on analysis
    manual_scale_factor : float or None
        Manually specify the scale factor between channels
        If None, will compute based on image statistics
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert from BGR (OpenCV default) to RGB for easier handling
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Split into RGB channels
    b_channel, g_channel, r_channel = cv2.split(img)  # OpenCV returns BGR
    
    # Save individual channels for inspection
    cv2.imwrite(f"{output_dir}/r_channel.png", r_channel)
    cv2.imwrite(f"{output_dir}/g_channel.png", g_channel)
    cv2.imwrite(f"{output_dir}/b_channel.png", b_channel)
    
    # -------------------------------------------------------------
    # STEP 1: Visualize histograms to help with manual adjustments
    # -------------------------------------------------------------
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(r_channel.ravel(), 256, [0, 256], color='red', alpha=0.7)
    plt.title('Red Channel Histogram')
    
    plt.subplot(1, 3, 2)
    plt.hist(g_channel.ravel(), 256, [0, 256], color='green', alpha=0.7)
    plt.title('Green Channel Histogram')
    
    plt.subplot(1, 3, 3)
    plt.hist(b_channel.ravel(), 256, [0, 256], color='blue', alpha=0.7)
    plt.title('Blue Channel Histogram')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/channel_histograms.png")
    
    # -------------------------------------------------------------
    # STEP 2: Select channels to work with
    # -------------------------------------------------------------
    channel_dict = {'R': r_channel, 'G': g_channel, 'B': b_channel}
    
    if manual_channel_selection:
        # Use manually specified channels
        channel1_name, channel2_name = manual_channel_selection
        channel1 = channel_dict[channel1_name]
        channel2 = channel_dict[channel2_name]
        print(f"Using manually selected channels: {channel1_name} and {channel2_name}")
    else:
        # Auto-select based on standard deviation (higher std = more information)
        std_values = {
            'R': np.std(r_channel),
            'G': np.std(g_channel),
            'B': np.std(b_channel)
        }
        
        # Find the two channels with highest standard deviation
        channels_sorted = sorted(std_values.items(), key=lambda x: x[1], reverse=True)
        channel1_name, _ = channels_sorted[0]
        channel2_name, _ = channels_sorted[1]
        
        channel1 = channel_dict[channel1_name]
        channel2 = channel_dict[channel2_name]
        
        print(f"Automatically selected channels based on standard deviation:")
        print(f"Channel STD values - R: {std_values['R']:.2f}, G: {std_values['G']:.2f}, B: {std_values['B']:.2f}")
        print(f"Selected: {channel1_name} and {channel2_name}")
    
    # -------------------------------------------------------------
    # STEP 3: Determine scaling factor
    # -------------------------------------------------------------
    if manual_scale_factor is not None:
        scale_factor = manual_scale_factor
        print(f"Using manually specified scale factor: {scale_factor}")
    else:
        # Calculate scale factor to match the means of the two channels
        # This is a starting point - you'll likely need to adjust this
        mean1 = np.mean(channel1)
        mean2 = np.mean(channel2)
        
        # Prevent division by zero
        if mean2 == 0:
            scale_factor = 1.0
        else:
            scale_factor = mean1 / mean2
        
        print(f"Calculated scale factor: {scale_factor:.4f}")
        print(f"Channel means - {channel1_name}: {mean1:.2f}, {channel2_name}: {mean2:.2f}")
    
    # -------------------------------------------------------------
    # STEP 4: Create multiple difference images with different scale factors
    # -------------------------------------------------------------
    # Create a range of scale factors around the calculated one
    if manual_scale_factor is None:
        scale_factors = [scale_factor * x for x in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]]
    else:
        scale_factors = [manual_scale_factor]
    
    for idx, sf in enumerate(scale_factors):
        # Apply scaling and calculate difference
        channel1_float = channel1.astype(np.float32)
        channel2_scaled = channel2.astype(np.float32) * sf
        
        # Try both subtraction directions
        diff1 = cv2.absdiff(channel1_float, channel2_scaled)
        diff2 = cv2.absdiff(channel2_scaled, channel1_float)
        
        # Normalize to 0-255
        diff1_norm = cv2.normalize(diff1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        diff2_norm = cv2.normalize(diff2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Save difference images
        cv2.imwrite(f"{output_dir}/diff_{channel1_name}_{channel2_name}_sf{sf:.2f}.png", diff1_norm)
        cv2.imwrite(f"{output_dir}/diff_{channel2_name}_{channel1_name}_sf{sf:.2f}.png", diff2_norm)
        
        # Create enhanced versions using contrast stretching
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        diff1_enhanced = clahe.apply(diff1_norm)
        diff2_enhanced = clahe.apply(diff2_norm)
        
        cv2.imwrite(f"{output_dir}/diff_enhanced_{channel1_name}_{channel2_name}_sf{sf:.2f}.png", diff1_enhanced)
        cv2.imwrite(f"{output_dir}/diff_enhanced_{channel2_name}_{channel1_name}_sf{sf:.2f}.png", diff2_enhanced)
        
        # Create inverted versions (sometimes better for visibility)
        cv2.imwrite(f"{output_dir}/diff_inv_{channel1_name}_{channel2_name}_sf{sf:.2f}.png", 255 - diff1_norm)
        cv2.imwrite(f"{output_dir}/diff_inv_{channel2_name}_{channel1_name}_sf{sf:.2f}.png", 255 - diff2_norm)
        
        # If it's the main scale factor, create a more comprehensive visualization
        if idx == len(scale_factors) // 2 or len(scale_factors) == 1:
            # Create a visualization of results
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 3, 1)
            plt.imshow(img_rgb)
            plt.title("Original Image")
            
            plt.subplot(2, 3, 2)
            plt.imshow(channel1, cmap='gray')
            plt.title(f"{channel1_name} Channel")
            
            plt.subplot(2, 3, 3)
            plt.imshow(channel2, cmap='gray')
            plt.title(f"{channel2_name} Channel")
            
            plt.subplot(2, 3, 4)
            plt.imshow(diff1_norm, cmap='gray')
            plt.title(f"Diff ({channel1_name} - {channel2_name}*{sf:.2f})")
            
            plt.subplot(2, 3, 5)
            plt.imshow(diff1_enhanced, cmap='gray')
            plt.title("Enhanced Difference")
            
            plt.subplot(2, 3, 6)
            plt.imshow(255 - diff1_norm, cmap='gray')
            plt.title("Inverted Difference")
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/comparison_sf{sf:.2f}.png")
    
    # -------------------------------------------------------------
    # STEP 5: Create interactive comparison of different scale factors (optional)
    # -------------------------------------------------------------
    if len(scale_factors) > 1:
        fig, axes = plt.subplots(len(scale_factors), 2, figsize=(12, 3*len(scale_factors)))
        
        for i, sf in enumerate(scale_factors):
            axes[i, 0].imshow(cv2.imread(f"{output_dir}/diff_{channel1_name}_{channel2_name}_sf{sf:.2f}.png"), cmap='gray')
            axes[i, 0].set_title(f"Scale Factor: {sf:.2f}")
            
            axes[i, 1].imshow(cv2.imread(f"{output_dir}/diff_enhanced_{channel1_name}_{channel2_name}_sf{sf:.2f}.png"), cmap='gray')
            axes[i, 1].set_title(f"Enhanced, Scale Factor: {sf:.2f}")
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/scale_factor_comparison.png")
    
    print("\nProcessing complete. Results saved to '{output_dir}' directory.")
    print("Examine the generated images to find the best scale factor and subtraction direction for your palimpsest.")
    print("Tips for getting better results:")
    print("1. Try different channel combinations (R-G, R-B, G-B)")
    print("2. Adjust the scale factor manually")
    print("3. Check both subtraction directions")
    print("4. Look at the enhanced and inverted versions")

if __name__ == "__main__":
    # Replace with your image path
    image_path = "palimpsest-2025.jpg"
    
    # Example 1: Let the algorithm choose channels automatically
    separate_palimpsest(image_path, output_dir="palimpsest_results_auto")
    
    # Example 2: Manually specify channels and scale factor
    # separate_palimpsest(
    #     image_path, 
    #     output_dir="palimpsest_results_manual",
    #     manual_channel_selection=('R', 'B'),  # Try different combinations
    #     manual_scale_factor=1.2  # Adjust this value
    # )

# Function to interactively test multiple parameters
def interactive_test(image_path, output_base_dir="interactive_tests"):
    """Test multiple channel combinations and scale factors"""
    channel_combinations = [('R', 'G'), ('R', 'B'), ('G', 'B')]
    scale_factors = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    for ch_combo in channel_combinations:
        for scale in scale_factors:
            output_dir = f"{output_base_dir}/{ch_combo[0]}_{ch_combo[1]}_sf{scale:.1f}"
            print(f"\nTesting {ch_combo[0]}-{ch_combo[1]} with scale factor {scale}")
            separate_palimpsest(
                image_path,
                output_dir=output_dir,
                manual_channel_selection=ch_combo,
                manual_scale_factor=scale
            )
    
    print("\nInteractive testing complete.")
    print(f"Check the '{output_base_dir}' directory to compare all combinations.")

# Uncomment to run interactive testing
# interactive_test("palimpsest.jpg")