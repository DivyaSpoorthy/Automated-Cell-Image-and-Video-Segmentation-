import cv2
import numpy as np
import os

PATH = "data_2"


def image_formats():
    return [".tif"]

class PreProcessing:

    def __init__(self, images):
        self.images = images

    def histogram_equalization(self):
        kernel = np.array([[-1, -1, -1], [-1, 3, -1], [-1, -1, -1]])
        images_after_eqalization = []
        for images in self.images:
            # clahe = cv2.createCLAHE(clipLimit=1000, tileGridSize=(10, 10))
            images_after_eqalization.append(cv2.equalizeHist(images))
        return np.array(images_after_eqalization)

    def three_dimensional_array_refactor(self, imgs):
        xDimension = []
        for i in range(imgs.shape[1]):
            yDimension = []
            for j in range(imgs.shape[2]):
                zDimension = []
                for k in imgs:
                    zDimension.append(k[i][j])
                yDimension.append(np.array(zDimension))
            xDimension.append(np.array(yDimension))

        reshaped = np.array(xDimension)
        return reshaped


    def cosine_similarity(self, a, b):
        dotProduct = abs(np.dot(a, b))
        mod_a = np.linalg.norm(a)
        mod_b = np.linalg.norm(b)
        denominator = mod_a * mod_b
        cos_similarity = dotProduct / denominator
        return cos_similarity

    def random_color(self):
        b = np.random.randint(255, size=(1,))
        g = np.random.randint(255, size=(1,))
        r = np.random.randint(255, size=(1,))

        Color = [b[0], g[0], r[0]]

        return np.array(Color)



def main():
    np.set_printoptions(threshold=np.inf)

    cell_images = []
    for image_format in image_formats():
        path = PATH
        image_file_list = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(image_format)]

        for image_file in image_file_list:
            cell_images.append(cv2.imread(image_file, 0))  # reading image in grey scale
    preprocessed_func = PreProcessing(cell_images)
    preprocessed_output = preprocessed_func.three_dimensional_array_refactor(np.array(cell_images))
    print(preprocessed_output.shape)
    print(len(cell_images))


if __name__ == "__main__":
    main()
