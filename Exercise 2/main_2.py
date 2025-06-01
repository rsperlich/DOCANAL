# Authors: Raphael Sperlich & Jonas Neumair
#
# Task B: OCR
#
# Goal: Text recognition using Tesseract
#
# Prerequisits
# -) Execute main_1.py
# 
# Execution
# python main_2.py --tesseract-all
#
# For single binarization or single folder tesseract, there are command line options

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
    """Returns 0 or 255 depending on the window high contrast pixels"""
    window_size = int(np.sqrt(len(window) / 2))
    i_image = window.reshape(2, window_size, window_size)[0]
    e_image = window.reshape(2, window_size, window_size)[1]

    I_pixel = i_image[1, 1]
    N_e = len(np.where(e_image<127)[0])
    E_mean = np.mean(i_image[e_image<127])
    E_std = np.std(i_image[e_image<127])
    # min number of high contrast pixel
    N_min = window_size - 1 # guessed value
    
    if (N_e >= N_min) and (I_pixel <= (E_mean + (E_std/2))):
        return 0
    return 255


def binarization(file_path: str) -> None:
    """Binarizes image based on Sue et al."""
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
    e_image = cv.imread(target_folder_file_path, cv.IMREAD_GRAYSCALE)
    hdt_image = generic_filter(np.stack((i_image, e_image)), historical_document_thresholding, size=(2, stroke_width, stroke_width))
    cv.imwrite(target_folder_file_path, hdt_image[1, :, :])


def run_tesseract() -> None:
    cropped_path = "cropped_images/"
    os.makedirs(cropped_path, exist_ok=True)
    os.makedirs(f"{cropped_path}binarized_otsu/", exist_ok=True)
    os.makedirs(f"{cropped_path}binarized_sue/", exist_ok=True)

    for file_name in sorted(os.listdir(cropped_path)):
        file_path = f"{cropped_path}{file_name}"
        if os.path.isdir(file_path):
            continue
        if file_name.startswith("."):
            continue


        txt_file_path = f"{cropped_path}txt_color/{file_name[:-4]}"
        print(f"tesseract {file_path} {txt_file_path}")
        os.makedirs(f"{cropped_path}txt_color/", exist_ok=True)
        os.system(f"tesseract {file_path} {txt_file_path}")

        # Otsu Binarize
        image = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
        _, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        file_path_binarized = f"{cropped_path}binarized_otsu/{file_name}"
        cv.imwrite(file_path_binarized, image)
        txt_file_path = f"{cropped_path}txt_binarized_otsu/{file_name[:-4]}"
        print(f"binarized otsu - tesseract {file_path_binarized} {txt_file_path}")
        os.makedirs(f"{cropped_path}txt_binarized_otsu/", exist_ok=True)
        os.system(f"tesseract {file_path_binarized} {txt_file_path}")

        # Sue Binarize
        binarization(file_path)
        file_path_binarized_sue = f"{cropped_path}binarized_sue/{file_name}"
        image = cv.imread(file_path_binarized_sue, cv.IMREAD_GRAYSCALE)
        txt_file_path = f"{cropped_path}txt_binarized_sue/{file_name[:-4]}"
        print(f"binarized sue - tesseract {file_path_binarized_sue} {txt_file_path}")
        os.makedirs(f"{cropped_path}txt_binarized_sue/", exist_ok=True)
        os.system(f"tesseract {file_path_binarized_sue} {txt_file_path}")

        # PSM using Sue et al.
        # PSM 1 - Automatic page segmentation with OSD.
        file_path_binarized_sue = f"{cropped_path}binarized_sue/{file_name}"
        txt_file_path = f"{cropped_path}txt_binarized_sue_psm1/{file_name[:-4]}"
        print(f"tesseract {file_path_binarized_sue} {txt_file_path} --psm 1")
        os.makedirs(f"{cropped_path}txt_binarized_sue_psm1/", exist_ok=True)
        os.system(f"tesseract {file_path_binarized_sue} {txt_file_path} --psm 1")

        # PSM 4 - Assume a single column of text of variable sizes.
        txt_file_path = f"{cropped_path}txt_binarized_sue_psm4/{file_name[:-4]}"
        print(f"tesseract {file_path_binarized_sue} {txt_file_path} --psm 4")
        os.makedirs(f"{cropped_path}txt_binarized_sue_psm4/", exist_ok=True)
        os.system(f"tesseract {file_path_binarized_sue} {txt_file_path} --psm 4")


    # Full Image
    full_path = "cropped_images/binarized_sue_full_images/"
    for file_name in sorted(os.listdir(full_path)):
        file_path = f"{full_path}{file_name}"
        if os.path.isdir(file_path):
            continue
        if file_name.startswith("."):
            continue

        # Default PSM is 3
        txt_file_path = f"{cropped_path}txt_full_images/{file_name[:-4]}"
        print(f"tesseract {file_path} {txt_file_path}")
        os.makedirs(f"{cropped_path}txt_full_images//", exist_ok=True)
        os.system(f"tesseract {file_path} {txt_file_path}")

        # PSM 0 - Orientation and script detection (OSD) only.
        txt_file_path = f"{cropped_path}txt_full_images_psm0/{file_name[:-4]}"
        print(f"tesseract {file_path} {txt_file_path} --psm 0")
        os.makedirs(f"{cropped_path}txt_full_images_psm0/", exist_ok=True)
        os.system(f"tesseract {file_path} {txt_file_path} --psm 0")

        # PSM 1 - Automatic page segmentation with OSD.
        txt_file_path = f"{cropped_path}txt_full_images_psm1/{file_name[:-4]}"
        print(f"tesseract {file_path} {txt_file_path} --psm 1")
        os.makedirs(f"{cropped_path}txt_full_images_psm1/", exist_ok=True)
        os.system(f"tesseract {file_path} {txt_file_path} --psm 1")

        # PSM 4 - Assume a single column of text of variable sizes.
        txt_file_path = f"{cropped_path}txt_full_images_psm4/{file_name[:-4]}"
        print(f"tesseract {file_path} {txt_file_path} --psm 4")
        os.makedirs(f"{cropped_path}txt_full_images_psm4/", exist_ok=True)
        os.system(f"tesseract {file_path} {txt_file_path} --psm 4")
        
        txt_file_path = f"{cropped_path}txt_full_images_psm6/{file_name[:-4]}"
        print(f"tesseract {file_path} {txt_file_path} --psm 6")
        os.makedirs(f"{cropped_path}txt_full_images_psm6/", exist_ok=True)
        os.system(f"tesseract {file_path} {txt_file_path} --psm 6")

        txt_file_path = f"{cropped_path}txt_full_images_psm11/{file_name[:-4]}"
        print(f"tesseract {file_path} {txt_file_path} --psm 11")
        os.makedirs(f"{cropped_path}txt_full_images_psm11/", exist_ok=True)
        os.system(f"tesseract {file_path} {txt_file_path} --psm 11")


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
            "filename": pred_file_path[pred_file_path.rfind('/')+1:-4], 
            "cer": cer_score, "wer": wer_score
        })

    return scores


def print_cer_wer_scores_all():
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
    for score in get_cer_wer_scores("cropped_images/txt_binarized_sue_psm1/"):
        cer_scores.append(score['cer'])
        wer_scores.append(score['wer'])
    print("Binarized Cropped Sue PSM 1")
    print(f"CER Average: {np.mean(cer_scores)}")
    print(f"WER Average: {np.mean(wer_scores)}")

    cer_scores = []
    wer_scores = []
    for score in get_cer_wer_scores("cropped_images/txt_binarized_sue_psm4/"):
        cer_scores.append(score['cer'])
        wer_scores.append(score['wer'])
    print("Binarized Cropped Sue PSM 4")
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
    print("Colored Full Images PSM 3 (Default)")
    print(f"CER Average: {np.mean(cer_scores)}")
    print(f"WER Average: {np.mean(wer_scores)}")

    cer_scores = []
    wer_scores = []
    for score in get_cer_wer_scores("cropped_images/txt_full_images_psm0/"):
        cer_scores.append(score['cer'])
        wer_scores.append(score['wer'])
    print("Colored Full Images PSM 0")
    print(f"CER Average: {np.mean(cer_scores)}")
    print(f"WER Average: {np.mean(wer_scores)}")

    cer_scores = []
    wer_scores = []
    for score in get_cer_wer_scores("cropped_images/txt_full_images_psm1/"):
        cer_scores.append(score['cer'])
        wer_scores.append(score['wer'])
    print("Colored Full Images PSM 1")
    print(f"CER Average: {np.mean(cer_scores)}")
    print(f"WER Average: {np.mean(wer_scores)}")

    cer_scores = []
    wer_scores = []
    for score in get_cer_wer_scores("cropped_images/txt_full_images_psm4/"):
        cer_scores.append(score['cer'])
        wer_scores.append(score['wer'])
    print("Colored Full Images PSM 4")
    print(f"CER Average: {np.mean(cer_scores)}")
    print(f"WER Average: {np.mean(wer_scores)}")

    cer_scores = []
    wer_scores = []
    for score in get_cer_wer_scores("cropped_images/txt_full_images_psm6/"):
        cer_scores.append(score['cer'])
        wer_scores.append(score['wer'])
    print("Colored Full Images PSM 6")
    print(f"CER Average: {np.mean(cer_scores)}")
    print(f"WER Average: {np.mean(wer_scores)}")

    cer_scores = []
    wer_scores = []
    for score in get_cer_wer_scores("cropped_images/txt_full_images_psm11/"):
        cer_scores.append(score['cer'])
        wer_scores.append(score['wer'])
    print("Colored Full Images PSM 11")
    print(f"CER Average: {np.mean(cer_scores)}")
    print(f"WER Average: {np.mean(wer_scores)}")

    cer_scores = []
    wer_scores = []
    for score in get_cer_wer_scores("cropped_images/txt_binarized_sue_full_images/"):
        cer_scores.append(score['cer'])
        wer_scores.append(score['wer'])
    print("Binarized Full Images PSM 4")
    print(f"CER Average: {np.mean(cer_scores)}")
    print(f"WER Average: {np.mean(wer_scores)}")


def print_cer_wer_scores(input_path):
    cer_scores = []
    wer_scores = []
    for score in get_cer_wer_scores(f"cropped_images/{input_path}/"):
        cer_scores.append(score['cer'])
        wer_scores.append(score['wer'])
    print("Score")
    print(f"CER Average: {np.mean(cer_scores)}")
    print(f"WER Average: {np.mean(wer_scores)}")


def tesseract(input_folder, output_folder, psm=3, oem=3):
    """Run tesseract path"""
    full_path = f"{input_folder}/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in sorted(os.listdir(full_path)):
        file_path = f"{full_path}{file_name}"
        if os.path.isdir(file_path):
            continue
        if file_name.startswith("."):
            continue

        # Default PSM is 3
        txt_file_path = f"{output_folder}/{file_name[:-4]}"
        print(f"tesseract {file_path} {txt_file_path} --psm {psm} --oem {oem}")
        os.system(f"tesseract {file_path} {txt_file_path} --psm {psm} --oem {oem}")


def binarize_sue(input_folder):
    """Binarized input folder"""
    for file_name in sorted(os.listdir(input_folder)):
        file_path = f"{input_folder}/{file_name}"
        if os.path.isdir(file_path):
            continue
        if file_name.startswith("."):
            continue
        
        binarization(file_path)


def binarize_otsu(input_folder):
    for file_name in sorted(os.listdir(input_folder)):
        file_path = f"{input_folder}/{file_name}"
        if os.path.isdir(file_path):
            continue
        if file_name.startswith("."):
            continue

        print(file_path)
        # Otsu Binarize
        image = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
        _, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        file_path_binarized = f"{input_folder}/binarized_otsu/{file_name}"
        cv.imwrite(file_path_binarized, image)


def print_report():
    for folder_name in sorted(os.listdir("cropped_images/")):
        if folder_name.startswith("."):
            continue
        if not folder_name.startswith("txt_"):
            continue
        folder_path = f"cropped_images/{folder_name}/"
        print(folder_path)
        if os.path.isdir(folder_path) and list(os.listdir(folder_path)):
            scores = get_cer_wer_scores(folder_path)
            print(scores[6-1])
            print(scores[23-1])
            print(scores[38-1])
            print(scores[40-1])
            print(scores[56-1])
            print(scores[63-1])
            print(scores[13-1])
            print(scores[61-1])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="Task B",
        description="Program to tesseract binarized and colored images"
    )

    parser.add_argument('-ea', '--evaluate-all', action='store_true')
    parser.add_argument('-ta', '--tesseract-all', action='store_true')
    parser.add_argument('-eo', '--evaluate-only')
    parser.add_argument('-to', '--tesseract-only', nargs=4)
    parser.add_argument('-bs', '--binarize-sue')
    parser.add_argument('-bo', '--binarize-otsu')
    parser.add_argument('-fs', '--folder-score')

    args = parser.parse_args()

    if args.evaluate_only:
        print(args)
        print_cer_wer_scores(args.evaluate_only)
        return
    
    if args.tesseract_only:
        psm = 3
        oem = 3
        if len(args.tesseract_only) >= 3:
            psm = args.tesseract_only[2]
        
        if len(args.tesseract_only) == 4:
            oem = args.tesseract_only[3]

        tesseract(args.tesseract_only[0], args.tesseract_only[1], psm, oem)
        return
    
    if args.binarize_sue:
        print(args.binarize_sue)
        binarize_sue(args.binarize_sue)

    if args.binarize_otsu:
        print(args.binarize_otsu)
        binarize_otsu(args.binarize_otsu)

    if args.evaluate_all:
        print_cer_wer_scores()
        return
    
    if args.tesseract_all:
        run_tesseract()
        return
    
    if args.folder_score:
        for file_score in get_cer_wer_scores(args.folder_score):
            print(file_score)
        return

if __name__ == "__main__":
    main()
    #print_report()


#{'filename': 'ed_images/txt_binarized_sue/crop_0006_ImageFile_{0F3BA1D5-D334-47C6-B148-F3DDDABB36B1}5', 'cer': 0.2222222222222222, 'wer': 1.6
#{'filename': 'ed_images/txt_binarized_sue/crop_0023_ImageFile_{37EC9D6D-28BF-4C21-A49B-2D971EDDDBA0}1', 'cer': 0.2916666666666667, 'wer': 0.9411764705882353}
#{'filename': 'ed_images/txt_binarized_sue/crop_0038_ImageFile_{279326F9-D817-4DC6-9D7E-76C6D62DD6DA}4', 'cer': 0.058823529411764705, 'wer': 0.4}
#{'filename': 'ed_images/txt_binarized_sue/crop_0040_ImageFile_{19270373-75C9-4FB6-8E80-35EBBA26513C}25', 'cer': 0.1038961038961039, 'wer': 2.6666666666666665}
#{'filename': 'ed_images/txt_binarized_sue/crop_0056_ImageFile_{E83C86EF-A0B2-477A-98AA-43BF86ABFB64}11', 'cer': 0.1590909090909091, 'wer': 0.75}
#{'filename': 'ed_images/txt_binarized_sue/crop_0063_ImageFile_{1CA6F88E-71EA-439A-8197-B1C9AE9458B2}3', 'cer': 0.5757575757575758, 'wer': 1.0}