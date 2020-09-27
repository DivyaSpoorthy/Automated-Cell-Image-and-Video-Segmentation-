import cv2
import numpy as np
import os
from collections import deque


# class DBSCAN:
#
#     def __init__(self, reshaped_image, radius, min_points):
#         self.reshaped_image = reshaped_image
#         self.radius = radius
#         self.min_points = min_points
#
#     def process(self, threeD_reshaped_image, epsilon, min_pts, intensity_threshold):
#         Label = np.zeros(threeD_reshaped_image[:, :, 0].shape, dtype=int)
#         C = 0
#         for i in range(threeD_reshaped_image.shape[0]):
#             for j in range(threeD_reshaped_image.shape[1]):
#
#                 if Label[i, j] == 1:
#                     continue
#                 dimension = (2 * epsilon) + 1
#                 neighbour_pix = []
#
#                 neighbour_count, neighbour_pix = self.neighbourhoodPoints(
#                         threeD_reshaped_image,
#                         epsilon, i, j, intensity_threshold)
#
#                 if neighbour_count < min_pts:
#                     Label[i][j] = -1
#                 C = C + 1
#                 Label[i][j] = C
#
#                 neighbour_pix.remove((i, j))
#                 for neighbour_pixel in neighbour_pix:
#                     if Label[neighbour_pixel[0]][neighbour_pixel[1]] == -1:
#                         Label[neighbour_pixel[0]][neighbour_pixel[1]] = C
#                     if Label[neighbour_pixel[0]][neighbour_pixel[1]] != 0:
#                         continue
#                     Label[neighbour_pixel[0]][neighbour_pixel[1]] = C
#                     expanded_neighbour_count, expanded_neighbours = self.neighbourhoodPoints(threeD_reshaped_image, epsilon, neighbour_pixel[0], neighbour_pixel[1], intensity_threshold)
#                     if expanded_neighbour_count >= min_pts:
#                         neighbour_pix.extend(expanded_neighbours)
#
#
#     def neighbourhoodPoints(self, matrix, eps, index1, index2, threshold):
#         matrix = np.array(matrix)
#         padded_matrix = np.pad(matrix, (eps, eps), 'constant')
#         print(padded_matrix)
#         n = (2 * eps) + 1
#         neighbour_cnt = 0
#         neighbour_pixels = []
#         comparing_pixel = padded_matrix[eps + index1][eps + index2]
#         for i in range(n):  # will always include the point in itself
#             for j in range(n):
#                 current_pixel = padded_matrix[i + index1][j + index2]
#                 print(current_pixel, comparing_pixel)
#                 print(comparing_pixel - current_pixel)
#                 diff = np.linalg.norm(current_pixel - comparing_pixel)
#                 print(diff)
#                 if diff < threshold:
#                     neighbour_cnt += 1
#
#         return neighbour_cnt


def process(threeD_reshaped_image, epsilon, min_pts, intensity_threshold):
    Label = np.zeros(threeD_reshaped_image[:, :, 0].shape, dtype=int)
    C = 0
    for i in range(threeD_reshaped_image.shape[0]):
        for j in range(threeD_reshaped_image.shape[1]):

            if Label[i, j] == 1:
                continue
            dimension = (2 * epsilon) + 1
            neighbour_pix = []

            neighbour_count, neighbour_pix = neighbourhoodPoints(
                threeD_reshaped_image,
                epsilon, i, j, intensity_threshold)

            if neighbour_count < min_pts:
                Label[i][j] = -1
            C = C + 1
            Label[i][j] = C

            neighbour_pix.remove((i, j))
            for neighbour_pixel in neighbour_pix:
                if Label[neighbour_pixel[0]][neighbour_pixel[1]] == -1:
                    Label[neighbour_pixel[0]][neighbour_pixel[1]] = C
                if Label[neighbour_pixel[0]][neighbour_pixel[1]] != 0:
                    continue
                Label[neighbour_pixel[0]][neighbour_pixel[1]] = C
                expanded_neighbour_count, expanded_neighbours = neighbourhoodPoints(threeD_reshaped_image, epsilon,
                                                                                    neighbour_pixel[0],
                                                                                    neighbour_pixel[1],
                                                                                    intensity_threshold)
                if expanded_neighbour_count >= min_pts:
                    neighbour_pix.extend(expanded_neighbours)

    cv2.imwrite("DBSCAN_1.png", Label)


def neighbourhoodPoints(matrix, eps, index1, index2, threshold):
    matrix = np.array(matrix)
    padded_matrix = np.pad(matrix, (eps, eps), 'constant', constant_values=(999999))
    print(padded_matrix)
    n = (2 * eps) + 1
    neighbours_cnt = 0
    neighbour_pixels = []
    comparing_pixel = padded_matrix[eps + index1][eps + index2]
    for i in range(n):  # will always include the point in itself
        for j in range(n):
            current_pixel = padded_matrix[i + index1][j + index2]
            print(current_pixel, comparing_pixel)
            print(comparing_pixel - current_pixel)
            diff = np.linalg.norm(current_pixel - comparing_pixel)
            print(diff)
            if diff < threshold:
                neighbours_cnt += 1
                neighbour_pixels.append((index1, index2))
    return neighbours_cnt, neighbour_pixels


testmatrix = np.random.randint(10, size=(3, 3, 2))
ans, pix = neighbourhoodPoints(testmatrix, 1, 1, 1, 1)
print(ans, pix)
