import cv2
import numpy as np


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


