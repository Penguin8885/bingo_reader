import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import module_img_analyzer as ia

if __name__ == '__main__':

    file_names = os.listdir("./data/3001-3200/")
    for file_name in file_names:
        print(file_name)
        img = cv2.imread("./data/3001-3200/" + file_name)

        threshold_img = ia.get_thresholding_img(img)
        rectcontour_img = ia.get_rectcontour_img(img, threshold_img)

        cv2.imwrite("./result/3001-3200/" + file_name, rectcontour_img)
