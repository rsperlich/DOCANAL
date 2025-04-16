import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_and_split_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    R, G, B = cv2.split(image_rgb)
    return R, G, B, image_rgb

def compute_redness_index(R, G):
    R_f = R.astype(np.float32)
    G_f = G.astype(np.float32)
    redness_index = (R_f - G_f) / (R_f + G_f + 1e-5)
    redness_index_norm = cv2.normalize(redness_index, None, 0, 255, cv2.NORM_MINMAX)
    redness_index_uint8 = redness_index_norm.astype(np.uint8)
    return redness_index_uint8

def threshold_redness_index(redness_index_uint8, percentile=80):
    threshold_value = int(np.percentile(redness_index_uint8, percentile))
    _, redness_thresh = cv2.threshold(redness_index_uint8, threshold_value, 255,
                                       cv2.THRESH_TOZERO)
    return redness_thresh, threshold_value

def find_intensity_threshold(channel, mask, channel_name=""):
    channel_text_values = channel[mask]
    hist, _ = np.histogram(channel_text_values, bins=256, range=(0, 256))
    hist_smooth = np.convolve(hist, np.ones(5) / 5, mode='same')
    derivative = np.diff(hist_smooth)
    local_mins = []
    for i in range(1, len(derivative)):
        if derivative[i - 1] < 0 and derivative[i] > 0:
            local_mins.append((i, hist_smooth[i]))
    valid_mins = [(pos, val) for pos, val in local_mins if 30 < pos < 220]
    if channel_name.lower() == "green":
        print(f"validmins (Green): {valid_mins}")
        if len(valid_mins) > 2:
            split_thresh = valid_mins[2][0]
        elif len(valid_mins) > 1:
            split_thresh = valid_mins[1][0]
        elif len(valid_mins) == 1:
            split_thresh = valid_mins[0][0]
        else:
            split_thresh = np.median(channel_text_values)
    elif valid_mins:
        split_thresh = valid_mins[0][0]
    else:
        split_thresh = np.median(channel_text_values)
    print(f"[{channel_name}] Intensity threshold: {split_thresh}")
    return split_thresh, hist_smooth, valid_mins

def separate_text_layers(R, G, B, mask_R, mask_G, mask_B, split_thresh_R,
                        split_thresh_G, split_thresh_B):
    overwriting_mask_r = (R < split_thresh_R) & mask_R
    underwriting_mask_r = (R >= split_thresh_R) & mask_R

    overwriting_r = np.zeros_like(R)
    overwriting_r[overwriting_mask_r] = R[overwriting_mask_r]
    underwriting_r = np.zeros_like(R)
    underwriting_r[underwriting_mask_r] = R[underwriting_mask_r]

    overwriting_mask_g = (G < split_thresh_G) & mask_G
    underwriting_mask_g = (G >= split_thresh_G) & mask_G

    overwriting_g = np.zeros_like(G)
    overwriting_g[overwriting_mask_g] = G[overwriting_mask_g]
    underwriting_g = np.zeros_like(G)
    underwriting_g[underwriting_mask_g] = G[underwriting_mask_g]

    overwriting_mask_b = (B < split_thresh_B) & mask_B
    underwriting_mask_b = (B >= split_thresh_B) & mask_B

    overwriting_b = np.zeros_like(B)
    overwriting_b[overwriting_mask_b] = B[overwriting_mask_b]
    underwriting_b = np.zeros_like(B)
    underwriting_b[underwriting_mask_b] = B[underwriting_mask_b]

    return overwriting_r, underwriting_r, overwriting_mask_r, underwriting_mask_r, overwriting_g, underwriting_g, overwriting_mask_g, underwriting_mask_g, overwriting_b, underwriting_b, overwriting_mask_b, underwriting_mask_b

def display_separated_texts(overwriting_r, underwriting_r):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Low Red Values (Overwriting)")
    plt.imshow(overwriting_r, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("High Red Values (Underwriting)")
    plt.imshow(underwriting_r, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def display_histogram(hist_smooth, split_thresh, channel, valid_mins=None):
    plt.figure(figsize=(12, 6))
    plt.plot(hist_smooth, color='blue')
    plt.axvline(x=split_thresh, color='red', linestyle='--')
    if valid_mins:
        for min_pos, min_val in valid_mins:
            plt.plot(min_pos, min_val, 'go')
    plt.title(f"Smoothed Histogram of {channel} Channel with Threshold = {split_thresh}")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()

def get_pixel_rgb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global R, G, B
        r = R[y, x]
        g = G[y, x]
        b = B[y, x]
        print(f"Pixel coordinates: ({x}, {y})  RGB values: ({r}, {g}, {b})")

def calculate_text_mask(R,G,B, r_thresh_low, r_thresh_high, g_thresh_low, g_thresh_high, b_thresh_low, b_thresh_high):
    return (R > r_thresh_low) & (R < r_thresh_high) & (G > g_thresh_low) & (G < g_thresh_high) & (B > b_thresh_low) & (B < b_thresh_high)

def main(image_path):
    global R, G, B
    print("Loading and splitting image...")
    R, G, B, image_rgb = load_and_split_image(image_path)

    print("Computing redness index for overwriting text...")
    redness_index_uint8 = compute_redness_index(R, G)

    redness_thresh, threshold_value = threshold_redness_index(
        redness_index_uint8)

    print("Finding intensity threshold for text separation...")
    mask_R = redness_thresh > 0
    split_thresh_R, hist_smooth_R, valid_mins_R = find_intensity_threshold(
        R, mask_R, channel_name="Red")
    split_thresh_G, hist_smooth_G, valid_mins_G = find_intensity_threshold(
        G, mask_R, channel_name="Green")
    split_thresh_B, hist_smooth_B, valid_mins_B = find_intensity_threshold(
        B, mask_R, channel_name="Blue")
    
    mask_G = mask_R.copy() #added
    mask_B = mask_R.copy() #added

    plt.figure(figsize=(12, 6))
    plt.plot(hist_smooth_R, color='red', label='Red Channel')
    plt.axvline(x=split_thresh_R, color='red', linestyle='--')
    plt.plot(hist_smooth_G, color='green', label='Green Channel')
    plt.axvline(x=split_thresh_G, color='green', linestyle='--')
    plt.plot(hist_smooth_B, color='blue', label='Blue Channel')
    plt.axvline(x=split_thresh_B, color='blue', linestyle='--')
    plt.title("Smoothed Histograms of RGB Channels")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    print("Separating text layers based on intensity threshold...")
    (overwriting_r, underwriting_r, overwriting_mask_r, underwriting_mask_r,
     overwriting_g, underwriting_g, overwriting_mask_g, underwriting_mask_g,
     overwriting_b, underwriting_b, overwriting_mask_b, underwriting_mask_b) = separate_text_layers(
        R, G, B, mask_R, mask_G, mask_B, split_thresh_R, split_thresh_G, split_thresh_B) # Pass mask_B

    display_separated_texts(overwriting_r, underwriting_r)
    #cv2.namedWindow('Original Image')
    #cv2.setMouseCallback('Original Image', get_pixel_rgb)
    #cv2.imshow('Original Image', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    print("Saving low red values (overwriting) as overwriting_text_vertical...")
    overwriting_text_vertical = overwriting_r.copy()
    cv2.imwrite('overwriting_text_vertical.jpg', overwriting_text_vertical)

    # Now, let's try to extract the horizontal text.
    print("Separating text layers for horizontal text...")
    (overwriting_r_horizontal, underwriting_r_horizontal, overwriting_mask_r_horizontal, underwriting_mask_r_horizontal,
     overwriting_g_horizontal, underwriting_g_horizontal, overwriting_mask_g_horizontal, underwriting_mask_g_horizontal,
     overwriting_b_horizontal, underwriting_b_horizontal, overwriting_mask_b_horizontal, underwriting_mask_b_horizontal) = separate_text_layers(
        R, G, B, ~mask_R, ~mask_G, ~mask_B, split_thresh_R, split_thresh_G, split_thresh_B)

    print("Saving high red values (underwriting) as overwriting_text_horizontal...")
    overwriting_text_horizontal = underwriting_r_horizontal.copy()
    cv2.imwrite('overwriting_text_horizontal.jpg', overwriting_text_horizontal)

    # New attempt to extract horizontal text
    r_thresh_low = 160
    r_thresh_high = 210
    g_thresh_low = 90
    g_thresh_high = 130
    b_thresh_low = 0
    b_thresh_high = 45
    horizontal_text_mask = calculate_text_mask(R,G,B, r_thresh_low, r_thresh_high, g_thresh_low, g_thresh_high, b_thresh_low, b_thresh_high)
    horizontal_text = np.zeros_like(R)
    horizontal_text[horizontal_text_mask] = R[horizontal_text_mask]
    cv2.imwrite('horizontal_text.jpg', horizontal_text)
    
    #show the horizontal text mask
    cv2.imwrite('horizontal_text_mask.jpg', horizontal_text_mask.astype(np.uint8) * 255)
    
    #Display the horizontal text mask
    plt.figure(figsize=(12, 6))
    plt.imshow(horizontal_text_mask, cmap='gray')
    plt.title("Horizontal Text Mask")
    plt.xlabel("Pixel Column")
    plt.ylabel("Pixel Row")
    plt.show()
    
    # Display the extracted horizontal text
    plt.figure(figsize=(12, 6))
    plt.imshow(horizontal_text, cmap='gray')
    plt.title("Extracted Horizontal Text")
    plt.xlabel("Pixel Column")
    plt.ylabel("Pixel Row")
    plt.show()


if __name__ == "__main__":
    image_path = 'palimpsest-2025.jpg'
    main(image_path)
