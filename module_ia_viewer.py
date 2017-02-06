import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def _show_thresholding_img(file_name):
    img = cv2.imread(file_name)

    #threshold_img = get_gray_thresholding_img(img, 80, 255)
    #threshold_img = get_bgr_thresholdimg_img(img, [200, 200, 200], [255, 255, 255])
    threshold_img = get_hsv_thresholding_img(img, [0, 0, 0], [255, 255, 60])

    threshold_img = cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2RGB)
    plt.imshow(threshold_img)
    plt.show()

def _show_noisy_contour_img(file_name):
    img = cv2.imread(file_name)

    threshold_img = get_hsv_thresholding_img(img, [30, 0, 100], [180, 100, 255])
    contour_img, _ = get_rect_contour_img(img, threshold_img)

    contour_img = cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)
    plt.imshow(contour_img)
    plt.show()

def _show_contour_img(file_name):
    img = cv2.imread(file_name)

    threshold_img = get_hsv_thresholding_img(img, [30, 0, 100], [180, 100, 255])
    _, contour_img = get_rect_contour_img(img, threshold_img)
    #contour_img = get_circle_contour_img(img, threshold_img)

    contour_img = cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)
    plt.imshow(contour_img)
    plt.show()


def _write_imgs(file_name, output_name):
    img = cv2.imread(file_name)

    threshold_img = get_hsv_thresholding_img(img, [30, 0, 100], [180, 100, 255])
    noisy_contour_img, contour_img = get_rect_contour_img(img, threshold_img)

    cv2.imwrite("0_" + output_name, threshold_img)
    cv2.imwrite("1_" + output_name, noisy_contour_img)
    cv2.imwrite("2_" + output_name, rect_contour_img)


if __name__ == '__main__':
    file_name = "./data/3001-3200/IMG_4877.jpg"
    _show_thresholding_img(file_name)
    #_show_noisy_contour_img(file_name)
    #_show_contour_img(file_name)
    #_write_imgs(file_name, "sample.jpg")
