import os
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.ndimage import generic_filter
from skimage.morphology import skeletonize

"""
Part 2

Implement two binarization methods.

Method 1: Su et al.
 - Dataset: DIBCO2009
 - Metrics: F-Score, pseudo F-Score, P, R, PSNR

Method 2: U-net-based binarization
 - Dataset: DIBCO competition (except DIBCO2009 as training dataset)
 - Metrics: F-Score, pseudo F-Score, P, R, PSNR

 Report:
 - results of binarization methods on DIBCO2009
 - show comparison
 - describe advantages and disadvantages of both methods
 - report sample images
"""

def contrast_image_construction(window) -> int:
    """Returns contrast image construction filter"""
    epsilon = 1e-10
    return int(255 * ((np.max(window) - np.min(window)) / (np.max(window) + np.min(window) + epsilon)))

def historical_document_thresholding(window) -> int:
    """Returns ..."""
    #middle = len(window)//2
    #i_image = window[:middle]
    #e_image = window[middle:]
    i_image = window.reshape(2, 3, 3)[0]
    e_image = window.reshape(2, 3, 3)[1]

    #I_pixel = i_image[len(i_image)//2]
    #E_mean = np.sum(i_image * (1-e_image)) / N_e
    #E_std = np.sqrt(np.sum(np.power(((i_image - E_mean) * (1-e_image)), 2) / 2))

    I_pixel = i_image[1, 1]
    N_e = np.sum(e_image)
    E_mean = np.mean(i_image[e_image==0])
    E_std = np.std(i_image[e_image==0])
    N_min = 3
    
    if (N_e >= N_min) and (I_pixel <= (E_mean + (E_std/2))):
        return 0
    return 255

def show_img(image, img_name: str="Image") -> None:
    """Displays image, closes windows on any key input"""
    cv.imshow(img_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def binarization(file_path: str) -> None:
    """Main"""
    # File and path naming
    file_name = file_path[file_path.rfind("/")+1:-4]
    target_folder_name = "part2_images/"
    target_folder_file_path = f"{target_folder_name}{file_name}"

    # Input image
    image = cv.imread(file_path)
    if not Path(f"{target_folder_file_path}_v0.jpg").exists():
        cv.imwrite(f"{target_folder_file_path}_v0.jpg", image)
    
    # 2.1 Contrast Image Construction
    cic_file_path = f"{target_folder_file_path}_v1.jpg"
    if not Path(cic_file_path).exists():
        assert image.shape[2] == 3
        channels = []
        for channel in range(3):
            channels.append(generic_filter(image[:, :, channel], contrast_image_construction, size=3))
            cic_image = np.stack(channels, axis=-1)
            #cic_image = generic_filter(image, contrast_image_construction, 3)
        assert cic_image.shape[2] == 3
        cv.imwrite(cic_file_path, cic_image)

    # 2.2 High Contrast Pixel Detection
    hcpd_file_path = f"{target_folder_file_path}_v2.jpg"
    if not Path(hcpd_file_path).exists():
        greyscale_image = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
        _, hcpd_image = cv.threshold(greyscale_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        assert len(hcpd_image.shape) == 2
        cv.imwrite(hcpd_file_path, hcpd_image)

    # 2.3 Historical Document Threshholding
    hdt_file_path =  f"{target_folder_file_path}_v3.jpg"
    if not Path(hdt_file_path).exists():
        i_image = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
        hcpd_image = cv.imread(hcpd_file_path, cv.IMREAD_GRAYSCALE)
        e_image = (hcpd_image > 0).astype(np.uint8) # E(x, y) is equal to 0 if the document image pixel is detected as a high contrast pixel
        hdt_image = generic_filter(np.stack((i_image, e_image)), historical_document_thresholding, size=(2, 3, 3))
        cv.imwrite(hdt_file_path, hdt_image[1, :, :])

    print("Done")
    pass

def test_metrics(folder_path: str) -> None:
    metric_list = []
    for file_name in os.listdir(folder_path):
        if not file_name.endswith("_v3.jpg"):
            continue
        metrics = image_metrics(f"{folder_path}{file_name}", f"{folder_path}{file_name[:-6]}gt.tif")
        print(metrics)
        metric_list.append(metrics)

    #show_img(image)
    #show_img(image_gt)
    pass

def image_metrics(image_path: str, image_gt_path: str) -> dict:
    """Compares metrics of local min max vs test image"""
    file_name = image_path[image_path.rfind("/")+1:]
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    image_gt = cv.imread(image_gt_path, cv.IMREAD_GRAYSCALE)

    image_binary = (image > 127).astype(np.uint8)
    image_gt_binary = (image_gt > 127).astype(np.uint8)

    tp = ((image_binary == 0) & (image_gt_binary == 0)).sum()
    tn = ((image_binary == 1) & (image_gt_binary == 1)).sum()
    fp = ((image_binary == 0) & (image_gt_binary == 1)).sum()
    fn = ((image_binary == 1) & (image_gt_binary == 0)).sum()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_measure = (2 * recall * precision) / (recall + precision)

    # pseudo f measure
    im_inv = 1 - image_gt  
    sk = skeletonize(im_inv)
    im_sk = np.ones_like(image_gt)
    im_sk[sk] = 0
    precall = tp / np.sum(1-im_sk)
    pfmeasure = (2*precall*precision)/(precall+precision)
    #p_f_measure = (2 * p - recall * precision) / (p - recall + precision)
    
    total_pixel = image_gt_binary.shape[0]*image_gt_binary.shape[1]
    mse = np.sum(np.power(image_gt - image, 2)) / total_pixel
    c = 1 # distance between foreground and background
    psnr = 10 * np.log10(np.power(c, 2) / mse)

    assert (tp + tn + fp + fn) == total_pixel

    return {"filename": file_name, 
            "fmeasure": f_measure, "pseudo-fmeasure": pfmeasure, 
            "precision": precision, "recall": recall, "psnr": psnr}

def main() -> None:
    """Main"""
    handwritten_folder_path = "../../dibco2009/DIBC02009_Test_images-handwritten/"
    printed_folder_path = "../../dibco2009/DIBCO2009_Test_images-printed/"

    folders = [handwritten_folder_path, printed_folder_path]
    for folder in folders:
        for file_name in os.listdir(folder):
            file_path = f"{folder}{file_name}"
            if file_name.endswith("_gt.tif"):
                image_gt = cv.imread(file_path)
                cv.imwrite(f"part2_images/{file_name}", image_gt)
                continue

            binarization(folder+file_name)

    test_metrics(f"part2_images/")
    


if __name__ == "__main__":
    main()
    import torch
    net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)

