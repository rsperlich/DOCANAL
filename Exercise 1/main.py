import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_and_split_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    R, G, B = cv2.split(image_rgb)
    cv2.imwrite('channel_R.jpg', R)
    cv2.imwrite('channel_G.jpg', G)
    cv2.imwrite('channel_B.jpg', B)
    return image_rgb, R, G, B

def display_channels(R, G, B):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title('Red Channel')
    plt.imshow(R, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title('Green Channel')
    plt.imshow(G, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title('Blue Channel')
    plt.imshow(B, cmap='gray')
    plt.tight_layout()
    plt.show()

def compute_redness_index(R, G):
    R_f = R.astype(np.float32)
    G_f = G.astype(np.float32)
    redness_index = (R_f - G_f) / (R_f + G_f + 1e-5)
    redness_index_norm = cv2.normalize(redness_index, None, 0, 255, cv2.NORM_MINMAX)
    redness_index_uint8 = redness_index_norm.astype(np.uint8)
    cv2.imwrite("redness_index.jpg", redness_index_uint8)
    return redness_index_uint8

def display_redness_index(redness_index_uint8):
    plt.imshow(redness_index_uint8, cmap="hot")
    plt.title("Redness Index (Overwriting Highlighted)")
    plt.axis("off")
    plt.show()

def threshold_redness_index(redness_index_uint8, percentile=80):
    threshold_value = int(np.percentile(redness_index_uint8, percentile))
    print(f"Using upper quartile threshold value: {threshold_value}")
    _, redness_thresh = cv2.threshold(redness_index_uint8, threshold_value, 255, cv2.THRESH_TOZERO)
    cv2.imwrite("redness_index_thresholded.jpg", redness_thresh)
    return redness_thresh, threshold_value

def display_thresholded_index(redness_thresh, threshold_value):
    plt.imshow(redness_thresh, cmap="hot")
    plt.title(f"Thresholded Redness Index (>= {threshold_value})")
    plt.axis("off")
    plt.show()

def find_intensity_threshold(R, mask):
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
    plt.figure(figsize=(12, 6))
    plt.plot(hist_smooth, color='blue')
    plt.axvline(x=split_thresh, color='red', linestyle='--')
    plt.title(f"Smoothed Histogram of Red Values in Text Regions (Threshold = {split_thresh})")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()

def create_color_visualization(R, overwriting_mask, underwriting_mask):
    color_visualization = np.zeros((R.shape[0], R.shape[1], 3), dtype=np.uint8)
    color_visualization[overwriting_mask] = [255, 0, 0]
    color_visualization[underwriting_mask] = [0, 0, 255]
    
    cv2.imwrite("text_layers_color_coded.jpg", cv2.cvtColor(color_visualization, cv2.COLOR_RGB2BGR))
    return color_visualization

def display_color_visualization(color_visualization):
    plt.figure(figsize=(10, 8))
    plt.imshow(color_visualization)
    plt.title("Color-Coded Text Layers (Red = Overwriting, Blue = Underwriting)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def extract_underwriting_by_channel_subtraction(R, G, B):
    # Convert to float for calculations
    R_f = R.astype(np.float32)
    G_f = G.astype(np.float32)
    B_f = B.astype(np.float32)
    
    # Step 1: Sample regions to determine scaling factors
    # For demonstration, we'll use an automatic approach to find the scaling factor
    # We'll try different scaling factors and choose the one that minimizes variance in background regions
    
    # Create a rough mask of the parchment (background) by thresholding the red channel
    # (assuming parchment is relatively bright)
    _, background_mask = cv2.threshold(R, 180, 255, cv2.THRESH_BINARY)
    
    # Erode to ensure we're getting pure background without text edges
    kernel = np.ones((5,5), np.uint8)
    background_mask = cv2.erode(background_mask, kernel, iterations=2)
    
    # Try different scaling factors for the green channel
    best_scale = 1.0
    min_variance = float('inf')
    
    scale_factors = np.linspace(0.7, 1.3, 20)
    for scale in scale_factors:
        # Scale green channel and subtract from red channel
        temp_diff = R_f - (G_f * scale)
        
        # Check variance in background regions
        variance = np.var(temp_diff[background_mask > 0])
        
        if variance < min_variance:
            min_variance = variance
            best_scale = scale
    
    print(f"Best scaling factor for green channel: {best_scale:.3f}")
    
    # Step 2: Apply the optimal scaling factor
    scaled_G = G_f * best_scale
    
    # Step 3: Calculate the difference image
    diff_image = R_f - scaled_G
    
    # Step 4: Normalize the difference image for display
    diff_norm = cv2.normalize(diff_image, None, 0, 255, cv2.NORM_MINMAX)
    diff_uint8 = diff_norm.astype(np.uint8)
    
    # Step 5: Enhance contrast
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_diff = clahe.apply(diff_uint8)
    
    # Step 6: Try blue channel as well
    # Sometimes blue vs. red channel subtraction works better
    scaled_B = B_f * best_scale
    diff_image_B = R_f - scaled_B
    diff_norm_B = cv2.normalize(diff_image_B, None, 0, 255, cv2.NORM_MINMAX)
    diff_uint8_B = diff_norm_B.astype(np.uint8)
    enhanced_diff_B = clahe.apply(diff_uint8_B)
    
    # Save the results
    cv2.imwrite("underwriting_channel_diff_RG.jpg", diff_uint8)
    cv2.imwrite("underwriting_enhanced_RG.jpg", enhanced_diff)
    cv2.imwrite("underwriting_channel_diff_RB.jpg", diff_uint8_B)
    cv2.imwrite("underwriting_enhanced_RB.jpg", enhanced_diff_B)
    
    return diff_uint8, enhanced_diff, diff_uint8_B, enhanced_diff_B, best_scale

def display_channel_subtraction_results(diff_uint8, enhanced_diff, diff_uint8_B, enhanced_diff_B):
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.title("R-G Channel Difference")
    plt.imshow(diff_uint8, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.title("Enhanced R-G Difference (CLAHE)")
    plt.imshow(enhanced_diff, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.title("R-B Channel Difference")
    plt.imshow(diff_uint8_B, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.title("Enhanced R-B Difference (CLAHE)")
    plt.imshow(enhanced_diff_B, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
def analyze_underwriting_histogram(enhanced_diff):
    # Create mask for areas with potential underwriting text
    # We'll threshold the enhanced difference image to find the darker regions
    _, text_mask = cv2.threshold(enhanced_diff, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Use the mask to extract pixel values
    text_values = enhanced_diff[text_mask > 0]
    
    # Compute histogram of these values
    hist, bins = np.histogram(text_values, bins=256, range=(0, 256))
    hist_smooth = np.convolve(hist, np.ones(5)/5, mode='same')
    
    # Display histogram
    plt.figure(figsize=(12, 6))
    plt.plot(hist_smooth, color='blue')
    plt.title("Histogram of Underwriting Text Pixels (Channel Subtraction Method)")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.show()
    
    return hist_smooth
def extract_underwriting_from_RB_difference(R, B):
    # Convert to float for calculations
    R_f = R.astype(np.float32)
    B_f = B.astype(np.float32)
    
    # Try different scaling factors for the blue channel
    best_scale = 1.0
    min_variance = float('inf')
    
    # Create a rough mask of the parchment (background)
    _, background_mask = cv2.threshold(R, 180, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    background_mask = cv2.erode(background_mask, kernel, iterations=2)
    
    # Find optimal scaling factor
    scale_factors = np.linspace(0.7, 1.3, 20)
    for scale in scale_factors:
        temp_diff = R_f - (B_f * scale)
        variance = np.var(temp_diff[background_mask > 0])
        if variance < min_variance:
            min_variance = variance
            best_scale = scale
    
    print(f"Optimal scaling factor for R-B difference: {best_scale:.3f}")
    
    # Apply the optimal scaling
    scaled_B = B_f * best_scale
    diff_image = R_f - scaled_B
    
    # Normalize to full range
    diff_norm = cv2.normalize(diff_image, None, 0, 255, cv2.NORM_MINMAX)
    diff_uint8 = diff_norm.astype(np.uint8)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_diff = clahe.apply(diff_uint8)
    
    # Separate underwriting text using adaptive thresholding
    # This works better than global thresholding for faint text
    binary_adaptive = cv2.adaptiveThreshold(
        enhanced_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, 4
    )
    
    # Remove noise with morphological operations
    kernel = np.ones((2,2), np.uint8)
    binary_cleaned = cv2.morphologyEx(binary_adaptive, cv2.MORPH_OPEN, kernel)
    
    # Create visual output
    underwriting_RB = np.zeros_like(R)
    underwriting_RB[binary_cleaned > 0] = R[binary_cleaned > 0]
    
    # Save outputs
    cv2.imwrite("underwriting_RB_diff.jpg", diff_uint8)
    cv2.imwrite("underwriting_RB_enhanced.jpg", enhanced_diff)
    cv2.imwrite("underwriting_RB_binary.jpg", binary_cleaned)
    cv2.imwrite("underwriting_RB_final.jpg", underwriting_RB)
    
    # Display results
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.title("R-B Difference")
    plt.imshow(diff_uint8, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.title("Enhanced R-B Difference")
    plt.imshow(enhanced_diff, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.title("Binary Thresholded")
    plt.imshow(binary_cleaned, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.title("Extracted Underwriting")
    plt.imshow(underwriting_RB, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return diff_uint8, enhanced_diff, binary_cleaned, underwriting_RB


def process_palimpsest(image_path):
    print("Loading and splitting image...")
    image_rgb, R, G, B = load_and_split_image(image_path)
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
    
    print("\nExtracting underwriting using channel subtraction method...")
    diff_RG, enhanced_RG, diff_RB, enhanced_RB, scale_factor = extract_underwriting_by_channel_subtraction(R, G, B)
    display_channel_subtraction_results(diff_RG, enhanced_RG, diff_RB, enhanced_RB)
    
    print("\nChannel subtraction analysis complete.")
    print(f"Optimal scaling factor found: {scale_factor:.3f}")
    print("Check the output images to determine which method reveals the underwriting most effectively.")
    
    underwriting_hist = analyze_underwriting_histogram(enhanced_RG)
    print("\nExtracting underwriting using R-B channel difference...")
    diff_RB, enhanced_RB, binary_RB, underwriting_RB = extract_underwriting_from_RB_difference(R, B)
    
    return {
        'red_channel': R,
        'redness_index': redness_index_uint8,
        'overwriting': overwriting,
        'underwriting': underwriting,
        'underwriting_channel_diff_RG': diff_RG,
        'underwriting_enhanced_RG': enhanced_RG,
        'underwriting_channel_diff_RB': diff_RB,
        'underwriting_enhanced_RB': enhanced_RB,
        'color_visualization': color_visualization,
        'underwriting_hist': underwriting_hist
    }

if __name__ == "__main__":
    results = process_palimpsest('palimpsest-2025.jpg')