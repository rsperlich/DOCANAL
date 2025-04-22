from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def separate_palimpsest(image_path, output_dir='results', manual_channel_selection=None, manual_scale_factor=None):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    b_channel, g_channel, r_channel = cv2.split(img)

    cv2.imwrite(f"{output_dir}/r_channel.png", r_channel)
    cv2.imwrite(f"{output_dir}/g_channel.png", g_channel)
    cv2.imwrite(f"{output_dir}/b_channel.png", b_channel)

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

    channel_dict = {'R': r_channel, 'G': g_channel, 'B': b_channel}

    if manual_channel_selection:
        channel1_name, channel2_name = manual_channel_selection
        channel1 = channel_dict[channel1_name]
        channel2 = channel_dict[channel2_name]
        print(f"Using manually selected channels: {channel1_name} and {channel2_name}")
    else:
        std_values = {
            'R': np.std(r_channel),
            'G': np.std(g_channel),
            'B': np.std(b_channel)
        }

        channels_sorted = sorted(std_values.items(), key=lambda x: x[1], reverse=True)
        channel1_name, _ = channels_sorted[0]
        channel2_name, _ = channels_sorted[1]

        channel1 = channel_dict[channel1_name]
        channel2 = channel_dict[channel2_name]

        print(f"Automatically selected channels based on standard deviation:")
        print(f"Channel STD values - R: {std_values['R']:.2f}, G: {std_values['G']:.2f}, B: {std_values['B']:.2f}")
        print(f"Selected: {channel1_name} and {channel2_name}")

    if manual_scale_factor is not None:
        scale_factor = manual_scale_factor
        print(f"Using manually specified scale factor: {scale_factor}")
    else:
        mean1 = np.mean(channel1)
        mean2 = np.mean(channel2)
        if mean2 == 0:
            scale_factor = 1.0
        else:
            scale_factor = mean1 / mean2

        print(f"Calculated scale factor: {scale_factor:.4f}")
        print(f"Channel means - {channel1_name}: {mean1:.2f}, {channel2_name}: {mean2:.2f}")

    if manual_scale_factor is None:
        scale_factors = [scale_factor * x for x in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]]
    else:
        scale_factors = [manual_scale_factor]

    for idx, sf in enumerate(scale_factors):
        channel1_float = channel1.astype(np.float32)
        channel2_scaled = channel2.astype(np.float32) * sf

        diff1 = cv2.absdiff(channel1_float, channel2_scaled)
        diff2 = cv2.absdiff(channel2_scaled, channel1_float)

        diff1_norm = cv2.normalize(diff1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        diff2_norm = cv2.normalize(diff2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        cv2.imwrite(f"{output_dir}/diff_{channel1_name}_{channel2_name}_sf{sf:.2f}.png", diff1_norm)
        cv2.imwrite(f"{output_dir}/diff_{channel2_name}_{channel1_name}_sf{sf:.2f}.png", diff2_norm)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        diff1_enhanced = clahe.apply(diff1_norm)
        diff2_enhanced = clahe.apply(diff2_norm)

        cv2.imwrite(f"{output_dir}/diff_enhanced_{channel1_name}_{channel2_name}_sf{sf:.2f}.png", diff1_enhanced)
        cv2.imwrite(f"{output_dir}/diff_enhanced_{channel2_name}_{channel1_name}_sf{sf:.2f}.png", diff2_enhanced)

        cv2.imwrite(f"{output_dir}/diff_inv_{channel1_name}_{channel2_name}_sf{sf:.2f}.png", 255 - diff1_norm)
        cv2.imwrite(f"{output_dir}/diff_inv_{channel2_name}_{channel1_name}_sf{sf:.2f}.png", 255 - diff2_norm)

        if idx == len(scale_factors) // 2 or len(scale_factors) == 1:
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

    if len(scale_factors) > 1:
        fig, axes = plt.subplots(len(scale_factors), 2, figsize=(12, 3*len(scale_factors)))

        for i, sf in enumerate(scale_factors):
            axes[i, 0].imshow(cv2.imread(f"{output_dir}/diff_{channel1_name}_{channel2_name}_sf{sf:.2f}.png"), cmap='gray')
            axes[i, 0].set_title(f"Scale Factor: {sf:.2f}")

            axes[i, 1].imshow(cv2.imread(f"{output_dir}/diff_enhanced_{channel1_name}_{channel2_name}_sf{sf:.2f}.png"), cmap='gray')
            axes[i, 1].set_title(f"Enhanced, Scale Factor: {sf:.2f}")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/scale_factor_comparison.png")


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
    cv2.imwrite("redness_index.jpg", redness_index_uint8)
    return redness_index_uint8

def threshold_redness_index(redness_index_uint8, percentile=80):
    threshold_value = int(np.percentile(redness_index_uint8, percentile))
    _, redness_thresh = cv2.threshold(redness_index_uint8, threshold_value, 255, cv2.THRESH_TOZERO)
    cv2.imwrite("redness_index_thresholded.jpg", redness_thresh)
    return redness_thresh, threshold_value

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

def create_color_visualization(R, overwriting_mask, underwriting_mask):
    color_visualization = np.zeros((R.shape[0], R.shape[1], 3), dtype=np.uint8)
    color_visualization[overwriting_mask] = [255, 0, 0]
    color_visualization[underwriting_mask] = [0, 0, 255]

    cv2.imwrite("text_layers_color_coded.jpg", cv2.cvtColor(color_visualization, cv2.COLOR_RGB2BGR))
    return color_visualization

def process_palimpsest(image_path, output_dir):
    R, G, B, _ = load_and_split_image(image_path)
    redness_index_uint8 = compute_redness_index(R, G)
    redness_thresh, threshold_value = threshold_redness_index(redness_index_uint8)
    mask = redness_thresh > 0
    split_thresh, hist_smooth = find_intensity_threshold(R, mask)
    overwriting, underwriting, _, _ = separate_text_layers(R, mask, split_thresh)
    color_visualization = create_color_visualization(R, (R < split_thresh) & mask, (R >= split_thresh) & mask)
    plt.figure()
    plt.plot(hist_smooth, color='blue')
    plt.axvline(x=split_thresh, color='red', linestyle='--')
    plt.title(f"Smoothed Histogram of Red Values (Threshold = {split_thresh})")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.savefig(f"{output_dir}/smoothed_histogram_with_threshold.png")
    cv2.imwrite(f"{output_dir}/overwriting_red.png", overwriting)
    cv2.imwrite(f"{output_dir}/underwriting_red.png", underwriting)
    cv2.imwrite(f"{output_dir}/color_coded_text_layers.png", cv2.cvtColor(color_visualization, cv2.COLOR_RGB2BGR))

    plt.figure()
    plt.hist(redness_index_uint8.ravel(), 256, [0, 256], color='red', alpha=0.7)
    plt.title('Redness Index Histogram')
    plt.xlabel('Redness Index Value')
    plt.ylabel('Frequency')
    plt.savefig(f"{output_dir}/redness_index_histogram.png")


def interactive_test(image_path, output_base_dir="interactive_tests"):
    channel_combinations = [('R', 'G'), ('R', 'B'), ('G', 'B')]
    scale_factors = [0.6,0.65,0.7,0.8, 0.9, 1.0, 1.1, 1.2]

    for ch_combo in channel_combinations:
        for scale in scale_factors:
            output_dir = f"{output_base_dir}/{ch_combo[0]}_{ch_combo[1]}_sf{scale:.1f}"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            print(f"\nTesting {ch_combo[0]}-{ch_combo[1]} with scale factor {scale}")
            separate_palimpsest(
                image_path,
                output_dir=output_dir,
                manual_channel_selection=ch_combo,
                manual_scale_factor=scale
            )
            process_palimpsest(image_path, output_dir)


if __name__ == "__main__":
    image_path = "palimpsest-2025.jpg"
    separate_palimpsest(image_path, output_dir="palimpsest_results_auto")
    #separate_palimpsest(
    #    image_path,
    #    output_dir="palimpsest_results_manual",
    #    manual_channel_selection=('B', 'G'),
    #    manual_scale_factor=0.65
    #)
    process_palimpsest(image_path, "palimpsest_results_auto")