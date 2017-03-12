import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import module_img_analyzer as ia


if __name__ == '__main__':

    file_names = os.listdir("./data/")
    for file_name in file_names:
        print(file_name, "\a")
        img = cv2.imread("./data/"+file_name)
        threshold_img = ia.get_hsv_thresholding_img(img, [30, 0, 100], [180, 100, 255])
        _, contours_img = ia.get_rect_contour_img(img, threshold_img)

        cv2.imwrite("./result/"+file_name, contours_img)
