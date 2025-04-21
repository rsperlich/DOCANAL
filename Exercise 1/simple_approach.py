from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_and_split_image(image_path):
    """
    Loads a color image and splits it into its RGB channels.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: A tuple containing the Red, Green, and Blue channels as NumPy arrays, and the original RGB image.
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    R, G, B = cv2.split(image_rgb)
    return R, G, B, image_rgb

def display_channels(R, G, B):
    """
    Displays the individual Red, Green, and Blue channels of an image.

    Args:
        R (NumPy array): Red channel.
        G (NumPy array): Green channel.
        B (NumPy array): Blue channel.
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title('Red Channel')
    plt.imshow(R, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title('Green Channel')
    plt.imshow(G, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title('Blue Channel')
    plt.imshow(B, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def compute_redness_index(R, G):
    """
    Computes a redness index from the Red and Green channels.

    Args:
        R (NumPy array): Red channel.
        G (NumPy array): Green channel.

    Returns:
        NumPy array: The redness index image.
    """
    R_f = R.astype(np.float32)
    G_f = G.astype(np.float32)
    redness_index = (R_f - G_f) / (R_f + G_f + 1e-5)
    redness_index_norm = cv2.normalize(redness_index, None, 0, 255, cv2.NORM_MINMAX)
    redness_index_uint8 = redness_index_norm.astype(np.uint8)
    cv2.imwrite("redness_index.jpg", redness_index_uint8)
    return redness_index_uint8

def display_redness_index(redness_index_uint8):
    """
    Displays the redness index image.

    Args:
        redness_index_uint8 (NumPy array): The redness index image.
    """
    plt.imshow(redness_index_uint8, cmap="hot")
    plt.title("Redness Index (Overwriting Highlighted)")
    plt.axis("off")
    plt.show()

def threshold_redness_index(redness_index_uint8, percentile=80):
    """
    Thresholds the redness index image.

    Args:
        redness_index_uint8 (NumPy array): The redness index image.
        percentile (int, optional): The percentile value for thresholding. Defaults to 80.

    Returns:
        tuple: The thresholded image and the threshold value.
    """
    threshold_value = int(np.percentile(redness_index_uint8, percentile))
    print(f"Using upper quartile threshold value: {threshold_value}")
    _, redness_thresh = cv2.threshold(redness_index_uint8, threshold_value, 255, cv2.THRESH_TOZERO)
    cv2.imwrite("redness_index_thresholded.jpg", redness_thresh)
    return redness_thresh, threshold_value

def display_thresholded_index(redness_thresh, threshold_value):
    """
    Displays the thresholded redness index image.

    Args:
        redness_thresh (NumPy array): The thresholded redness index image.
        threshold_value (int): The threshold value.
    """
    plt.imshow(redness_thresh, cmap="hot")
    plt.title(f"Thresholded Redness Index (>= {threshold_value})")
    plt.axis("off")
    plt.show()

def find_intensity_threshold(R, mask):
    """
    Finds an intensity threshold to separate overwriting and underwriting text.

    Args:
        R (NumPy array): Red channel.
        mask (NumPy array): Mask indicating text regions.

    Returns:
        tuple: The intensity threshold and the smoothed histogram.
    """
    red_text_values = R[mask]
    hist, bins = np.histogram(red_text_values, bins=256, range=(0, 256))
    hist_smooth = np.convolve(hist, np.ones(5)/5, mode='same')

    local_mins = []
    for i in range(1, len(hist_smooth)-1):
        if hist_smooth[i-1] > hist_smooth[i] < hist_smooth[i+1]:
            local_mins.append((i, hist_smooth[i]))

    valid_mins = [(pos, val) for pos, val in local_mins
                    if val < np.mean(hist_smooth) and 30 < pos < 220]

    if valid_mins:
        valid_mins.sort(key=lambda x: x[1])
        split_thresh = valid_mins[0][0]
    else:
        split_thresh = np.median(red_text_values)

    print(f"Intensity threshold between two writings: {split_thresh}")
    return split_thresh, hist_smooth

def separate_text_layers(R, mask, split_thresh):
    """
    Separates the Red channel into overwriting and underwriting layers based on a threshold.

    Args:
        R (NumPy array): Red channel.
        mask (NumPy array): Mask indicating text regions.
        split_thresh (int): Intensity threshold.

    Returns:
        tuple: Overwriting and underwriting images, and their corresponding masks.
    """
    overwriting_mask = (R < split_thresh) & mask
    underwriting_mask = (R >= split_thresh) & mask

    overwriting = np.zeros_like(R)
    overwriting[overwriting_mask] = R[overwriting_mask]
    underwriting = np.zeros_like(R)
    underwriting[underwriting_mask] = R[underwriting_mask]

    cv2.imwrite("overwriting_red.jpg", overwriting)
    cv2.imwrite("underwriting_red.jpg", underwriting)

    return overwriting, underwriting, overwriting_mask, underwriting_mask

def display_separated_texts(overwriting, underwriting):
    """
    Displays the separated overwriting and underwriting text layers.

    Args:
        overwriting (NumPy array): Overwriting text layer.
        underwriting (NumPy array): Underwriting text layer.
    """
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Overwriting (Darker Red)")
    plt.imshow(overwriting, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("Underwriting (Lighter Red)")
    plt.imshow(underwriting, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def display_histogram(hist_smooth, split_thresh):
    """
    Displays the histogram of red values in the text regions, along with the threshold.

    Args:
        hist_smooth (NumPy array): Smoothed histogram.
        split_thresh (int): Intensity threshold.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(hist_smooth, color='blue')
    plt.axvline(x=split_thresh, color='red', linestyle='--')
    plt.title(f"Smoothed Histogram of Red Values in Text Regions (Threshold = {split_thresh})")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()

def create_color_visualization(R, overwriting_mask, underwriting_mask):
    """
    Creates a color visualization of the separated text layers.

    Args:
        R (NumPy array): Red channel.
        overwriting_mask (NumPy array): Mask for overwriting text.
        underwriting_mask (NumPy array): Mask for underwriting text.

    Returns:
        NumPy array: Color visualization image.
    """
    color_visualization = np.zeros((R.shape[0], R.shape[1], 3), dtype=np.uint8)
    color_visualization[overwriting_mask] = [255, 0, 0]  # Red for overwriting
    color_visualization[underwriting_mask] = [0, 0, 255]  # Blue for underwriting

    cv2.imwrite("text_layers_color_coded.jpg", cv2.cvtColor(color_visualization, cv2.COLOR_RGB2BGR))
    return color_visualization

def display_color_visualization(color_visualization):
    """
    Displays the color visualization of the separated text layers.

    Args:
        color_visualization (NumPy array): Color visualization image.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(color_visualization)
    plt.title("Color-Coded Text Layers (Red = Overwriting, Blue = Underwriting)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def process_palimpsest(image_path):
    """
    Processes a palimpsest image to separate overwriting and underwriting text.

    Args:
        image_path (str): Path to the palimpsest image.

    Returns:
        dict: A dictionary containing the processed results.
    """
    print("Loading and splitting image...")
    R, G, B, image_rgb = load_and_split_image(image_path)
    display_channels(R, G, B)

    print("Computing redness index for overwriting text...")
    redness_index_uint8 = compute_redness_index(R, G)
    display_redness_index(redness_index_uint8)

    redness_thresh, threshold_value = threshold_redness_index(redness_index_uint8)
    display_thresholded_index(redness_thresh, threshold_value)

    print("Finding intensity threshold for text separation...")
    mask = redness_thresh > 0
    split_thresh, hist_smooth = find_intensity_threshold(R, mask)

    print("Separating text layers based on intensity threshold...")
    overwriting, underwriting, overwriting_mask, underwriting_mask = separate_text_layers(R, mask, split_thresh)
    display_separated_texts(overwriting, underwriting)

    display_histogram(hist_smooth, split_thresh)

    color_visualization = create_color_visualization(R, overwriting_mask, underwriting_mask)
    display_color_visualization(color_visualization)

    return {
        'red_channel': R,
        'redness_index': redness_index_uint8,
        'overwriting': overwriting,
        'underwriting': underwriting,
        'color_visualization': color_visualization
    }

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

    results = process_palimpsest('palimpsest-2025.jpg')
