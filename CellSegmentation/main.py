import cv2
import numpy as np
import os

PATH = "data_2"


def image_formats():
    return [".png"]

class PreProcessing:

    def __init__(self, images):
        self.images = images

    def histogram_equalization(self):
        kernel = np.array([[-1, -1, -1], [-1, 3, -1], [-1, -1, -1]])
        images_after_eqalization = []
        for images in self.images:
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


class DBSCAN:

    def process(self, threeD_reshaped_image, epsilon, min_pts, intensity_threshold):
        Label = np.zeros(threeD_reshaped_image[:, :, 0].shape, dtype=int)
        C = 0
        for i in range(threeD_reshaped_image.shape[0]):
            print("itr:" + str(i))
            for j in range(threeD_reshaped_image.shape[1]):

                if Label[i][j] >= 1:
                    continue
                # print(i, j)
                dimension = (2 * epsilon) + 1
                neighbour_pix = []

                neighbour_count, neighbour_pix = self.neighbourhoodPoints(
                    threeD_reshaped_image,
                    epsilon, i, j, intensity_threshold)

                if neighbour_count < min_pts:
                    Label[i][j] = -1
                    continue
                C = C + 30
                Label[i][j] = C

                if (i, j) in neighbour_pix:
                    neighbour_pix.remove((i, j))
                for neighbour_pixel in neighbour_pix:

                    if Label[neighbour_pixel[0]][neighbour_pixel[1]] == -1:
                        Label[neighbour_pixel[0]][neighbour_pixel[1]] = C
                    if Label[neighbour_pixel[0]][neighbour_pixel[1]] != 0:
                        continue
                    Label[neighbour_pixel[0]][neighbour_pixel[1]] = C
                    expanded_neighbour_count, expanded_neighbours = self.neighbourhoodPoints(threeD_reshaped_image, epsilon,
                                                                                        neighbour_pixel[0],
                                                                                        neighbour_pixel[1],
                                                                                        intensity_threshold)
                    if expanded_neighbour_count >= min_pts:
                        neighbour_pix.extend(expanded_neighbours)

        cv2.imwrite("DBSCAN_1.png", Label)

    def neighbourhoodPoints(self, matrix, eps, index1, index2, threshold):
        matrix = np.array(matrix)
        padded_matrix = np.pad(matrix, (eps, eps), 'constant', constant_values=(999999))

        n = (2 * eps) + 1
        neighbours_cnt = 0
        neighbour_pixels = []
        comparing_pixel = padded_matrix[eps + index1][eps + index2]
        for i in range(n):  # will always include the point in itself
            for j in range(n):
                current_pixel = padded_matrix[index1 + i][index2 + j]

                diff = np.linalg.norm(current_pixel - comparing_pixel)

                if diff < threshold:
                    neighbours_cnt += 1
                    neighbour_pixels.append((i + index1 - eps, j + index2 - eps))
        return neighbours_cnt, neighbour_pixels




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

    dbscan = DBSCAN()
    dbscan.process(preprocessed_output, 1, 5, 0.1)










if __name__ == "__main__":
    main()
