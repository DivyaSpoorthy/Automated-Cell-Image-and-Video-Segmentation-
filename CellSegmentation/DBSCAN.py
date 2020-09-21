import cv2
import numpy as np
import os
from collections import deque


class DBSCAN:

    def __init__(self, reshaped_image, radius, min_points):
        self.reshaped_image = reshaped_image
        self.radius = radius
        self.min_points = min_points

    def process(self):
        L = np.zeros(self.reshaped_image[:, :, 0].shape, dtype=int)
        count = 0
        for i in range(self.reshaped_image.shape[0]):
            for j in range(self.reshaped_image.shape[1]):
                count += 1
                if L[i, j] == 1:
                    continue
                queue = deque([(i, j)])
                seen = []
                while queue:
                    coord = queue.popleft()
                    (x, y) = coord
                    if L[x, y] == 1:
                        continue
                    seen.append(coord)
                    L[x, y] = 1
                    epsilon = 1

    def core(self, epsilon, coordx, coordy, zeros_arr, que):
        ix = -1 * epsilon
        while (ix <= epsilon):
            jx = -1 * epsilon
            while (jx <= epsilon):
                if (coordx + ix >= 0 and coordy + jx >= 0 and coordx + ix <= self.reshaped_image.shape[0] - 1 and coordy + jx <= self.reshaped_image.shape[1] - 1 and zeros_arr[coordx + ix, coordy + jx] == 0):
                    px = self.reshaped_image[coordx + ix, coordy + jx]
                    # dist = abs(D[x,y] - px)
                    dist = np.linalg.norm(self.reshaped_image[coordx, coordy] - px)
                    # print(D[x,y], px, dist)
                    if (dist <= self.radius):
                        que.append((coordx + ix, coordy + jx))
                jx = jx + 1
            ix = ix + 1



