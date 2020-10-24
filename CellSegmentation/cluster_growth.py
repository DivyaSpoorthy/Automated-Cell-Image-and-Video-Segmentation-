import cv2
import numpy as np
import os

imageformat = ".png"

path = "data3_radius_images"
imfilelist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(imageformat)]

Images = []
# for el in imfilelist:
#     image = cv2.imread(el, 0)[:512, :512]
#     Images.append(image)

for el in range(10, 40):
    image = cv2.imread("DBSCAN_data1_output_rad_" + str(el) + ".png", 0)
    Images.append(image)
# cv2.imwrite("DBSCAN_tf.png", Images[0])
print((len(Images)))

def random_color():
    b = np.random.randint(255, size=(1,))
    g = np.random.randint(255, size=(1,))
    r = np.random.randint(255, size=(1,))

    color = [b[0], g[0], r[0]]

    return np.array(color)


def cluster_growth(image_time_frames, fixed_time_gap):
    image_time_frames.insert(0, np.zeros(image_time_frames[0][:, :].shape, dtype=int))
    Label = np.zeros((image_time_frames[0].shape[0], image_time_frames[0].shape[1], 3))
    Freeze = np.zeros((image_time_frames[0].shape[0], image_time_frames[0].shape[1]))
    for index in range(1, len(image_time_frames) - 8):

        print("index" + str(index))
        all_clusters_in_current_time_frame = {}

        for i in range(image_time_frames[index].shape[0]):  # getting all the clusters in current time frame
            for j in range(image_time_frames[index].shape[1]):
                if image_time_frames[index][i][j] == 0:
                    continue

                if image_time_frames[index][i][j] in all_clusters_in_current_time_frame:
                    all_clusters_in_current_time_frame[image_time_frames[index][i][j]].append((i, j))
                else:
                    all_clusters_in_current_time_frame[image_time_frames[index][i][j]] = [(i, j)]

        new_clusters_in_current_time_frame = []
        print(len(all_clusters_in_current_time_frame))

        previous_time_frame_of_image = image_time_frames[index - 1]
        future_time_frame = image_time_frames[index + fixed_time_gap]

        all_clusters_in_future_time_frame = {}
        for ix in range(future_time_frame.shape[0]):
            for jx in range(future_time_frame.shape[1]):
                if future_time_frame[ix][jx] == 0:
                    continue

                if future_time_frame[ix][jx] in all_clusters_in_future_time_frame:
                    all_clusters_in_future_time_frame[future_time_frame[ix][jx]].append((ix, jx))
                else:
                    all_clusters_in_future_time_frame[future_time_frame[ix][jx]] = [(ix, jx)]

        cv2.imwrite("fist_img.png", image_time_frames[index - 1])
        cv2.imwrite("future.png", image_time_frames[index + fixed_time_gap])
        cv2.imwrite("curr.png", image_time_frames[index])
        cnt = 0
        for i in all_clusters_in_current_time_frame:
            # print(i)
            cnt += 1
            print("one tf:" + str(cnt))
            cluster_indices = all_clusters_in_current_time_frame[i]


            is_new_cluster = 1
            for indice in cluster_indices:
                if previous_time_frame_of_image[indice[0]][indice[1]] != 0:
                    is_new_cluster = 0
                    break
            print(len(cluster_indices), is_new_cluster)
            if is_new_cluster:


                for cluster_in_future in all_clusters_in_future_time_frame:
                    cluster_inf = all_clusters_in_future_time_frame[cluster_in_future]
                    if len(cluster_inf) > 10000:
                        continue
                    # if set(cluster_indices).issubset(set(cluster_inf)):
                    if len(set(cluster_indices).intersection(set(cluster_inf))) > len(cluster_indices) // 2:
                        print("true")
                        color = random_color()
                        for elements in cluster_inf:
                            # if Freeze[elements[0]][elements[1]] == 1:
                            #     continue
                            # Freeze[elements[0]][elements[1]] = 1
                            Label[elements[0]][elements[1]][0] = color[0]
                            Label[elements[0]][elements[1]][1] = color[1]
                            Label[elements[0]][elements[1]][2] = color[2]
                        break

        cv2.imwrite("DBSCAN_data_1_timeframe_sizerestrict_"+str(index)+".png", Label)

    cv2.imwrite("DBSCAN_cluster_selection.png", Label)


cluster_growth(Images, 4)
