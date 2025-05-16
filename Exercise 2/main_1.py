# Authors: Raphael Sperlich & Jonas Neumair
#
# Task A: Obejct Detection Model (YoloV8)
#
# Goal: Extract the printed text of the images


import os
import numpy as np
import cv2 as cv
from ultralytics import YOLO


def show_image(image, imgage_name: str="Image") -> None:
    """Displays image, closes windows on any key input"""
    cv.imshow(imgage_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main() -> None:
    """Find machine printed text and save it to new file"""
    model = YOLO("yolov8n.pt")
    dataset_path = "dataset/"

    filepaths = []
    for file_name in os.listdir(dataset_path):
        file_path = f"{dataset_path}{file_name}"
        if os.path.isdir(file_path):
            continue
        filepaths.append(file_path)

    # Run the model on all files
    results = model(filepaths)

    for result, file_path in zip(results, filepaths):
        file_name = file_path[file_path.rfind('/')+1:]
        print(f"Read file {file_name}")
        boxes = result.boxes
        cropped_image = None
        image = cv.imread(file_path)
        for xyxy, cls in zip(boxes.xyxy, boxes.cls):
            # Only draw the rectangle around the machine printed text
            if result.names[int(cls)] == 'pt':
                #cv.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 10)
                if cropped_image is None:
                    cropped_image = image[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                    continue
                
                image = image[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                if cropped_image.shape[1] == image.shape[1]:
                    cv.vconcat([cropped_image, image])
                    continue

                # add to blank image to match both shapes
                if cropped_image.shape[1] > image.shape[1]:
                    padded_image = np.full((image.shape[0], cropped_image.shape[1], image.shape[2]), (255, 255, 255), dtype=np.uint8)
                    padded_image[0:image.shape[0], 0:image.shape[1]] = image
                    cropped_image = cv.vconcat([cropped_image, padded_image])
                    
        else:
            print(f"Writing image {file_name}")
            cv.imwrite(f"cropped_images/crop_{file_name}", cropped_image)
        

if __name__ == "__main__":
    main()
