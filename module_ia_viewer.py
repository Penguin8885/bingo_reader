import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import module_img_analyzer as ia


def show_thresholding_img(file_name):
    img = cv2.imread(file_name)

    #threshold_img = ia.get_canny_img(img, 10, 50)
    #threshold_img = ia.get_gray_thresholding_img(img, 80, 255)
    #threshold_img = ia.get_bgr_thresholdimg_img(img, [200, 200, 200], [255, 255, 255])
    threshold_img = ia.get_hsv_thresholding_img(img, [30, 0, 100], [180, 100, 255])

    threshold_img = cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2RGB)
    plt.imshow(threshold_img)
    plt.show()

def show_noisy_contour_img(file_name):
    img = cv2.imread(file_name)

    threshold_img = ia.get_hsv_thresholding_img(img, [30, 0, 100], [180, 100, 255])
    contour_img, _ = ia.get_rect_contour_img(img, threshold_img)

    contour_img = cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)
    plt.imshow(contour_img)
    plt.show()

def show_contour_img(file_name):
    img = cv2.imread(file_name)

    threshold_img = ia.get_hsv_thresholding_img(img, [30, 0, 100], [180, 100, 255])
    _, contour_img = ia.get_rect_contour_img(img, threshold_img)
    #contour_img = ia.get_circle_contour_img(img, threshold_img)

    contour_img = cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)
    plt.imshow(contour_img)
    plt.show()


def write_imgs(file_name, output_name):
    img = cv2.imread(file_name)

    threshold_img = ia.get_hsv_thresholding_img(img, [30, 0, 100], [180, 100, 255])
    noisy_contour_img, contour_img = ia.get_rect_contour_img(img, threshold_img)

    cv2.imwrite("0_" + output_name, threshold_img)
    cv2.imwrite("1_" + output_name, noisy_contour_img)
    cv2.imwrite("2_" + output_name, rect_contour_img)


if __name__ == '__main__':
    #4879
    #4874,4877,4881
    file_name = "./data/3001-3200/IMG_4874.jpg"
    #show_thresholding_img(file_name)
    #show_noisy_contour_img(file_name)
    show_contour_img(file_name)
    #write_imgs(file_name, "sample.jpg")
