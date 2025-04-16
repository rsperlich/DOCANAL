import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_and_split_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    R, G, B = cv2.split(image_rgb)
    return R, G, B, image_rgb

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
    
    return {
        'red_channel': R,
        'redness_index': redness_index_uint8,
        'overwriting': overwriting,
        'underwriting': underwriting,
        'color_visualization': color_visualization
    }

if __name__ == "__main__":
    results = process_palimpsest('palimpsest-2025.jpg')