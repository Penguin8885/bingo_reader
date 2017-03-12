import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def GaussianBlur(img, size=15):
    return cv2.GaussianBlur(img, (size, size), 0)
def medianBlur(img, size=3):
    return cv2.medianBlur(img, size)



def get_gray_thresholding_img(bgr_img, lower, upper, pre_blur_func=GaussianBlur, post_blur_func=None, blur_size=15):
    if pre_blur_func is not None:
        bgr_img = pre_blur_func(bgr_img, blur_size)

    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    _, threshold_img = cv2.threshold(gray_img, lower, upper, cv2.THRESH_BINARY)

    if post_blur_func is not None:
        threshold_img = post_blur_func(threshold_img, blur_size)

    return threshold_img

def get_bgr_thresholdimg_img(bgr_img, lower, upper, pre_blur_func=GaussianBlur, post_blur_func=None, blur_size=15):
    if pre_blur_func is not None:
        bgr_img = pre_blur_func(bgr_img, blur_size)

    threshold_img = cv2.inRange(
        bgr_img,
        np.array(lower, np.uint8),
        np.array(upper, np.uint8)
    )
    threshold_img = cv2.bitwise_not(threshold_img)

    if post_blur_func is not None:
        threshold_img = post_blur_func(threshold_img, blur_size)

    return threshold_img

def get_hsv_thresholding_img(bgr_img, lower, upper, pre_blur_func=GaussianBlur, post_blur_func=None, blur_size=15):
    if pre_blur_func is not None:
        bgr_img = pre_blur_func(bgr_img, blur_size)

    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    threshold_img = cv2.inRange(
        hsv_img,
        np.array(lower, np.uint8),
        np.array(upper, np.uint8)
    )
    threshold_img = cv2.bitwise_not(threshold_img)

    if post_blur_func is not None:
        threshold_img = post_blur_func(threshold_img, blur_size)

    return threshold_img

def get_canny_img(bgr_img, lower, upper, pre_blur_func=GaussianBlur, post_blur_func=None, blur_size=15):
    if pre_blur_func is not None:
        bgr_img = pre_blur_func(bgr_img, blur_size)

    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    canny_img = cv2.Canny(gray_img, lower, upper)

    if post_blur_func is not None:
        canny_img = post_blur_func(canny_img, blur_size)

    return canny_img



def get_circle_contour_img(bgr_img, binary_img):
    circles = cv2.HoughCircles(binary_img, cv2.HOUGH_GRADIENT, dp=5, minDist=400, minRadius=180, maxRadius=200)

    circle_contour_img = np.copy(bgr_img)
    if circles is not None and len(circles) > 0:
        for (x, y, r) in circles[0]:
            print((x,y,r))
            x = int(x); y = int(y); r = int(r)
            cv2.circle(circle_contour_img, (x,y), r, (0,255,0), 10)
            cv2.rectangle(circle_contour_img, (int(x-r),int(y-r)), (int(x+r),int(y+r)), (0,0,255), 10)
    else:
        print("cannot detect circles")

    return circle_contour_img

def get_rect_contour_img(bgr_img, binary_img):
    _, contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rect_contour_img = np.copy(bgr_img)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(rect_contour_img, (x,y), (x+w,y+h), (0,0,255), 10)

    return rect_contour_img
