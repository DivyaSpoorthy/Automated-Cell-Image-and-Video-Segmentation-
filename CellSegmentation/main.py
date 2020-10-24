import cv2
import numpy as np
import os

PATH = "data_2"


def image_formats():
    return [".tif"]

class PreProcessing:

    def __init__(self, images):
        self.images = images

    def histogram_equalization(self, imgs):
        kernel = np.array([[-1, -1, -1], [-1, 3, -1], [-1, -1, -1]])
        images_after_eqalization = []
        for images in imgs:
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




class DBSCAN:

    def __init__(self, threeDImage, epsilon):
        self.threeDImage = threeDImage
        self.epsilon = epsilon


    def precomputeIndices(self):
        print("precomputation start....")
        matrix = np.array(self.threeDImage)
        padded_matrix = np.pad(matrix, (self.epsilon, self.epsilon), 'constant', constant_values=(999999))
        d = {}
        for index1 in range(self.threeDImage.shape[0]):
            print("precomputed_indices:" + str(index1))
            for index2 in range(self.threeDImage.shape[1]):



                n = (2 * self.epsilon) + 1
                neighbours_cnt = 0
                neighbour_pixels = []
                comparing_pixel = padded_matrix[self.epsilon + index1][self.epsilon + index2]
                for i in range(n):  # will always include the point in itself
                    for j in range(n):
                        if ((i + index1 - self.epsilon >= self.threeDImage.shape[0] or j + index2 - self.epsilon >=
                             self.threeDImage.shape[1]) or (i + index1 - self.epsilon < 0 or j + index2 - self.epsilon < 0)):
                            continue

                        neighbour_pixels.append((i + index1 - self.epsilon, j + index2 - self.epsilon))
                d[(index1, index2)] = (neighbour_pixels)
                # return neighbours_cnt, neighbour_pixels
        print("precomutation ended !")
        return d



    def process(self, threeD_reshaped_image, epsilon, min_pts, intensity_threshold, precomputedValues, pca_image):
        Label = np.zeros(threeD_reshaped_image[:, :, 0].shape, dtype=int)
        # preComputedValues = self.precomputeIndices(threeD_reshaped_image, epsilon)
        C = 0
        number_of_clusters = 0
        all_clusters = []
        for i in range(threeD_reshaped_image.shape[0]):
            print("itr:" + str(i))
            for j in range(threeD_reshaped_image.shape[1]):

                if Label[i][j] >= 1:
                    continue
                # print(i, j)
                dimension = (2 * epsilon) + 1
                neighbour_pix = []

                # neighbour_count, neighbour_pix = self.neighbourhoodPoints(
                #     threeD_reshaped_image,
                #     epsilon, i, j, intensity_threshold)

                # neighbour_count, neighbour_pix = preComputedValues[(i, j)]
                neighbour_count, neighbour_pix = self.neighboursUnderIntensityThreshold(precomputedValues[(i,j)], i, j, threeD_reshaped_image, intensity_threshold, pca_image)

                if neighbour_count < min_pts:
                    Label[i][j] = -1
                    continue
                C = C + 30
                number_of_clusters += 1
                all_clusters.append(C)
                Label[i][j] = C

                if (i, j) in neighbour_pix:
                    neighbour_pix.remove((i, j))
                for neighbour_pixel in neighbour_pix:
                    if ((neighbour_pixel[0] >= threeD_reshaped_image.shape[0] or neighbour_pixel[1] >= threeD_reshaped_image.shape[1])
                            or (neighbour_pixel[0] < 0 or neighbour_pixel[1] < 0)):
                        continue
                    if Label[neighbour_pixel[0]][neighbour_pixel[1]] == -1:
                        Label[neighbour_pixel[0]][neighbour_pixel[1]] = C
                    if Label[neighbour_pixel[0]][neighbour_pixel[1]] != 0:
                        continue
                    Label[neighbour_pixel[0]][neighbour_pixel[1]] = C
                    # expanded_neighbour_count, expanded_neighbours = self.neighbourhoodPoints(threeD_reshaped_image, epsilon,
                    #                                                                     neighbour_pixel[0],
                    #                                                                     neighbour_pixel[1],
                    #                                                                     intensity_threshold)

                    expanded_neighbour_count, expanded_neighbours = self.neighboursUnderIntensityThreshold(
                        precomputedValues[(neighbour_pixel[0],neighbour_pixel[1])], neighbour_pixel[0], neighbour_pixel[1], threeD_reshaped_image, intensity_threshold, pca_image)



                    if expanded_neighbour_count >= min_pts:
                        neighbour_pix.extend(expanded_neighbours)

        print(number_of_clusters)
        cv2.imwrite("DBSCAN_test_2.png", Label)
        return Label, all_clusters

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
                # print(diff)
                if diff < threshold:
                    neighbours_cnt += 1
                    neighbour_pixels.append((i + index1 - eps, j + index2 - eps))
        return neighbours_cnt, neighbour_pixels

    def preComputation(self, threeDImage, eps, threshold):
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
                        # print(index1 + i, index2 + j)
                        current_pixel = padded_matrix[index1 + i][index2 + j]

                        diff = np.linalg.norm(current_pixel - comparing_pixel)
                        # print(d``iff)
                        if diff < threshold:
                            neighbours_cnt += 1
                            neighbour_pixels.append((i + index1 - eps, j + index2 - eps))
                d[(index1, index2)] = (neighbours_cnt, neighbour_pixels)
                # return neighbours_cnt, neighbour_pixels
        print("precomutation ended !")
        return d


    def neighboursUnderIntensityThreshold(self, neighbours, iCurr, jCurr, image, threshold, pca_image):

        comparing_pixel = image[iCurr][jCurr]
        neighbour_cnt = 0
        neighbour_pixels = []
        for neighbour in neighbours:
            current_pixel = pca_image[neighbour[0]][neighbour[1]]

            diff = np.linalg.norm(comparing_pixel/255.0 - current_pixel/255.0)
            # print(diff, threshold)
            if diff < threshold:
                neighbour_cnt += 1
                neighbour_pixels.append((neighbour[0], neighbour[1]))
        return neighbour_cnt, neighbour_pixels

class postprocessing:

    def random_color(self):
        b = np.random.randint(255, size=(1,))
        g = np.random.randint(255, size=(1,))
        r = np.random.randint(255, size=(1,))

        Color = [b[0], g[0], r[0]]

        return np.array(Color)

    def process(self, dbscan_output, clusters, image_number):
        cluster_color_map = {}
        for cluster in clusters:
            cluster_color_map[cluster] = self.random_color()

        color_dbscan_output = np.zeros((dbscan_output.shape[0], dbscan_output.shape[1], 3))
        for i in range(dbscan_output.shape[0]):
            for j in range(dbscan_output.shape[1]):
                if dbscan_output[i][j] in cluster_color_map:
                    random_color = cluster_color_map[dbscan_output[i][j]]
                    color_dbscan_output[i][j][0] = random_color[0]
                    color_dbscan_output[i][j][1] = random_color[1]
                    color_dbscan_output[i][j][2] = random_color[2]
        cv2.imwrite("DBSCAN_3D_intensity_threshold_" + str(image_number) + ".png", color_dbscan_output)
        return


def iterate_for_multiple_variable_params(preprocessed_image_output):

    post_processing = postprocessing()
    dbscan = DBSCAN(preprocessed_image_output, 4)
    precomputedValues = dbscan.precomputeIndices()
    thresholds = np.linspace(0,1,11)
    pca_image = cv2.imread("pca_comp1.png", 0)
    for intensity_threshold in range(5, 100):
        print("intensity:" + str(intensity_threshold))
        grey_output, cluster_list = dbscan.process(preprocessed_image_output, 4, 60, intensity_threshold/100.0, precomputedValues, pca_image)
        post_processing.process(grey_output, cluster_list, intensity_threshold)



def main():
    np.set_printoptions(threshold=np.inf)

    cell_images = []
    for image_format in image_formats():
        path = PATH
        image_file_list = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(image_format)]

        for image_file in image_file_list:
            cell_images.append(cv2.imread(image_file, 0))  # reading image in grey scale

    cell_images = [cell_images[0]]
    preprocessed_func = PreProcessing(cell_images)
    histogram_output = preprocessed_func.histogram_equalization(cell_images)
    cv2.imwrite("DBSCAN_original.png", histogram_output[0])
    # cell_images = np.divide(np.array(cell_images) , 255.0)
    preprocessed_output = preprocessed_func.three_dimensional_array_refactor(np.array(histogram_output))
    # preprocessed_output = np.divide(np.array(preprocessed_output), 255.0)
    print(preprocessed_output.shape)

    iterate_for_multiple_variable_params(preprocessed_output)










if __name__ == "__main__":
    main()
