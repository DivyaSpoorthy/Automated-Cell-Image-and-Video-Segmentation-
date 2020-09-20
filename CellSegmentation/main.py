import cv2
import numpy as np
import os

PATH = "data_2"


def image_formats():
    return [".tif"]


def main():
    np.set_printoptions(threshold=np.inf)

    cell_images = []
    for image_format in image_formats():
        path = PATH
        image_file_list = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(image_format)]

        for image_file in image_file_list:
            cell_images.append(cv2.imread(image_file, 0))  # reading image in grey scale
    print(len(cell_images))


if __name__ == "__main__":
    main()
