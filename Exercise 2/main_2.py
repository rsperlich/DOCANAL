# Authors: Raphael Sperlich & Jonas Neumair
#
# Task B: OCR
#
# Goal: Text recognition using Tesseract


import os
import numpy as np
import cv2 as cv
from scipy.ndimage import generic_filter
from jiwer import cer, wer

def show_img(image, img_name: str="Image") -> None:
    """Displays image, closes windows on any key input"""
    cv.imshow(img_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def contrast_image_construction(window) -> int:
    """Returns contrast image construction filter"""
    epsilon = 1e-20
    return 255* ((np.max(window) - np.min(window)) / (np.max(window) + np.min(window) + epsilon))


def historical_document_thresholding(window) -> int:
    """Returns ..."""
    window_size = int(np.sqrt(len(window) / 2))
    i_image = window.reshape(2, window_size, window_size)[0]
    e_image = window.reshape(2, window_size, window_size)[1]

    I_pixel = i_image[1, 1]
    N_e = np.sum(e_image)
    E_mean = np.mean(i_image[e_image==0])
    E_std = np.std(i_image[e_image==0])
    # min number of high contrast pixel
    N_min = window_size # guessed value
    
    if (N_e >= N_min) and (I_pixel <= (E_mean + (E_std/2))):
        return 0
    return 255


def binarization(file_path: str) -> None:
    """Main"""
    # File and path naming
    file_name = file_path[file_path.rfind("/")+1:-4]
    target_folder_name = "cropped_images/binarized_sue/"
    target_folder_file_path = f"{target_folder_name}{file_name}.jpg"

    # Input image in grayscale
    image = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
    
    # 2.1 Contrast Image Construction
    cic_image = generic_filter(image, contrast_image_construction, [3,3])

    # 2.2 High Contrast Pixel Detection using Otsu's global thresholding method
    _, hcpd_image = cv.threshold(cic_image, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    cv.imwrite(target_folder_file_path, hcpd_image)

    stroke_width = 3

    # 2.3 Historical Document Threshholding
    i_image = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
    hcpd_image = cv.imread(target_folder_file_path, cv.IMREAD_GRAYSCALE)
    e_image = (hcpd_image > 0).astype(np.uint8) # E(x, y) is equal to 0 if the document image pixel is detected as a high contrast pixel
    hdt_image = generic_filter(np.stack((i_image, e_image)), historical_document_thresholding, size=(2, stroke_width, stroke_width))
    cv.imwrite(target_folder_file_path, hdt_image[1, :, :])


def run_tesseract() -> None:
    cropped_path = "cropped_images/"

    for file_name in sorted(os.listdir(cropped_path)):
        file_path = f"{cropped_path}{file_name}"
        if os.path.isdir(file_path):
            continue

        txt_file_path = f"{cropped_path}txt_color/{file_name[:-4]}"
        print(f"tesseract {file_path} {txt_file_path}")
        os.system(f"tesseract {file_path} {txt_file_path}")

        # Otsu Binarize
        image = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
        _, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        file_path_binarized = f"{cropped_path}binarized_otsu/{file_name}"
        cv.imwrite(file_path_binarized, image)
        txt_file_path = f"{cropped_path}txt_binarized_otsu/{file_name[:-4]}"
        print(f"binarized otsu - tesseract {file_path_binarized} {txt_file_path}")
        os.system(f"tesseract {file_path_binarized} {txt_file_path}")

        # Sue Binarize
        binarization(file_path)
        file_path_binarized_sue = f"{cropped_path}binarized_sue/{file_name}"
        image = cv.imread(file_path_binarized_sue, cv.IMREAD_GRAYSCALE)
        txt_file_path = f"{cropped_path}txt_binarized_sue/{file_name[:-4]}"
        print(f"binarized sue - tesseract {file_path_binarized_sue} {txt_file_path}")
        os.system(f"tesseract {file_path_binarized_sue} {txt_file_path}")

    # Full Image
    full_path = "dataset/"
    for file_name in sorted(os.listdir(full_path)):
        file_path = f"{full_path}{file_name}"
        if os.path.isdir(file_path):
            continue

        # Default PSM is 3
        txt_file_path = f"{cropped_path}txt_full_images/{file_name[:-4]}"
        print(f"tesseract {file_path} {txt_file_path}")
        os.system(f"tesseract {file_path} {txt_file_path}")

        # PSM 0 - Orientation and script detection (OSD) only.
        txt_file_path = f"{cropped_path}txt_full_images_psm0/{file_name[:-4]}"
        print(f"tesseract {file_path} {txt_file_path} --psm 0")
        os.system(f"tesseract {file_path} {txt_file_path} --psm 0")

        # PSM 1 - Automatic page segmentation with OSD.
        txt_file_path = f"{cropped_path}txt_full_images_psm1/{file_name[:-4]}"
        print(f"tesseract {file_path} {txt_file_path} --psm 1")
        os.system(f"tesseract {file_path} {txt_file_path} --psm 1")

        # PSM 4 - Assume a single column of text of variable sizes.
        txt_file_path = f"{cropped_path}txt_full_images_psm4/{file_name[:-4]}"
        print(f"tesseract {file_path} {txt_file_path} --psm 4")
        os.system(f"tesseract {file_path} {txt_file_path} --psm 4")


def get_cer_wer_scores(path: str, ref_path: str="dataset/txt/"):
    scores = []
    for pred_file_name, ref_file_name in zip(sorted(os.listdir(path)), sorted(os.listdir(ref_path))):
        pred_file_path = f"{path}{pred_file_name}"
        ref_file_path = f"{ref_path}{ref_file_name}"

        pred_file_str = open(pred_file_path).read()
        ref_file_str = open(ref_file_path).read()

        cer_score = cer(pred_file_str, ref_file_str) if pred_file_str != "" else 1
        wer_score = wer(pred_file_str, ref_file_str) if pred_file_str != "" else 1

        scores.append({
            "filename": pred_file_path[5:-4], 
            "cer": cer_score, "wer": wer_score
        })

    return scores


def print_cer_wer_scores():
    cer_scores = []
    wer_scores = []
    for score in get_cer_wer_scores("cropped_images/txt_binarized_otsu/"):
        cer_scores.append(score['cer'])
        wer_scores.append(score['wer'])
    print("Binarized Cropped Otsu")
    print(f"CER Average: {np.mean(cer_scores)}")
    print(f"WER Average: {np.mean(wer_scores)}")

    cer_scores = []
    wer_scores = []
    for score in get_cer_wer_scores("cropped_images/txt_binarized_sue/"):
        cer_scores.append(score['cer'])
        wer_scores.append(score['wer'])
    print("Binarized Cropped Sue")
    print(f"CER Average: {np.mean(cer_scores)}")
    print(f"WER Average: {np.mean(wer_scores)}")

    cer_scores = []
    wer_scores = []
    for score in get_cer_wer_scores("cropped_images/txt_color/"):
        cer_scores.append(score['cer'])
        wer_scores.append(score['wer'])
    print("Colored Cropped Images")
    print(f"CER Average: {np.mean(cer_scores)}")
    print(f"WER Average: {np.mean(wer_scores)}")

    cer_scores = []
    wer_scores = []
    for score in get_cer_wer_scores("cropped_images/txt_full_images/"):
        cer_scores.append(score['cer'])
        wer_scores.append(score['wer'])
    print("Colored Full Images")
    print(f"CER Average: {np.mean(cer_scores)}")
    print(f"WER Average: {np.mean(wer_scores)}")

    cer_scores = []
    wer_scores = []
    for score in get_cer_wer_scores("cropped_images/txt_full_images_psm/"):
        cer_scores.append(score['cer'])
        wer_scores.append(score['wer'])
    print("Colored Full Images")
    print(f"CER Average: {np.mean(cer_scores)}")
    print(f"WER Average: {np.mean(wer_scores)}")


def main() -> None:
    import sys
    if "--evaluate-only" in sys.argv:
        print_cer_wer_scores()
        return
    
    if "--tesseract-only" in sys.argv:
        run_tesseract()
        return

    run_tesseract()
    print_cer_wer_scores()


if __name__ == "__main__":
    main()
