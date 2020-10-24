import cv2
import numpy as np


class preprocessing:

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








def preComputation(threeDImage, eps, threshold):
    d = {}
    for index1 in range(threeDImage.shape[0]):
        print("intr" + str(index1))
        for index2 in range(threeDImage.shape[1]):

            matrix = np.array(threeDImage)
            padded_matrix = np.pad(matrix, (eps, eps), 'constant', constant_values=(999999))

            n = (2 * eps) + 1
            neighbours_cnt = 0
            neighbour_pixels = []
            comparing_pixel = padded_matrix[eps + index1][eps + index2]
            for i in range(n):  # will always include the point in itself
                for j in range(n):
                    current_pixel = padded_matrix[index1 + i][index2 + j]

                    diff = np.linalg.norm(current_pixel - comparing_pixel)
                    # print(d``iff)
                    if diff < threshold:
                        neighbours_cnt += 1
                        neighbour_pixels.append((i + index1 - eps, j + index2 - eps))
            d[(index1, index2)] = [neighbours_cnt, neighbour_pixels]
            # return neighbours_cnt, neighbour_pixels
    return d






def precomputeIndices(threeDImage, eps):
    print("precomputation start....")

    d = {}
    for index1 in range(threeDImage.shape[0]):
        print("precomputed:" + str(index1))
        for index2 in range(threeDImage.shape[1]):

            matrix = np.array(threeDImage)
            padded_matrix = np.pad(matrix, (eps, eps), 'constant', constant_values=(999999))

            n = (2 * eps) + 1
            neighbours_cnt = 0
            neighbour_pixels = []
            comparing_pixel = padded_matrix[eps + index1][eps + index2]
            for i in range(n):  # will always include the point in itself
                for j in range(n):
                    if ((i + index1 - eps >= threeDImage.shape[0] or j + index2 - eps >=
                         threeDImage.shape[1])or (i + index1 - eps < 0 or j + index2 - eps < 0)):
                        continue

                    neighbour_pixels.append((i + index1 - eps, j + index2 - eps))
            d[(index1, index2)] = (neighbour_pixels)
            # return neighbours_cnt, neighbour_pixels
    print("precomutation ended !")
    return d


def neighboursUnderIntensityThreshold(neighbours, iCurr, jCurr, image, threshold):

    comparing_pixel = image[iCurr][jCurr]
    neighbour_cnt = 0
    neighbour_pixels = []
    for neighbour in neighbours:
        current_pixel = image[neighbour[0]][neighbour[1]]

        diff = np.linalg.norm(current_pixel - comparing_pixel)
        # print(d``iff)
        if diff < threshold:
            neighbour_cnt += 1
            neighbour_pixels.append((neighbour[0], neighbour[1]))
    return neighbour_cnt, neighbour_pixels

# img = cv2.imread("pca_comp5.png",0)[:5, :5]
# print(np.array(img).shape)
# res = precomputeIndices(np.array(img), 1)
# print(res)

def superpositionWithPCA(pca_image, original_image, threshold):

    for i in range(pca_image.shape[0]):
        for j in range(pca_image.shape[1]):
            # print(pca_image[i][j])
            if pca_image[i][j] <= threshold:
                original_image[i][j] = 0
    return original_image

def intensityMeasuresOfCentralTendency(image, median_area):
    cnt = 0
    intensity_mean = 0
    median_array = []
    median_set = set()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            cnt += 1
            intensity_mean += image[i][j]
            median_array.append(image[i][j])
            median_set.add(image[i][j])
    print(median_set)
    median_number = int(cnt * median_area)
    median = median_array[median_number]
    mean = intensity_mean/cnt
    return mean, median


pca = cv2.imread("data_2_pca_comp1.png", 0)
original = cv2.imread("data_2_original.tif", 0)

Mean, Median = intensityMeasuresOfCentralTendency(pca, 0.8)
print(Mean, Median)

res = superpositionWithPCA(np.array(pca), np.array(original), Median)
cv2.imwrite("pca_superposed_data_2_median.png", res)

