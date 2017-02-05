import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def get_thresholding_img(img):
    blur_img = cv2.GaussianBlur(img, (15,15), 0)

    hsv_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2HSV)
    threshold_img = cv2.inRange(
        hsv_img,
        np.array([30,0,100], np.uint8),
        np.array([180,100,255], np.uint8)
    )

    threshold_img = cv2.bitwise_not(threshold_img)

    return threshold_img

def get_rectcontour_img(img, binary_img):
    _, contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnt_info_tuple_list = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w/h
        area = w*h
        if (ratio > 0.7 and ratio < 1.3) and (area > 80000 and area < 500000):
            cnt_info_tuple_list.append((x, y, x+w, y+h))

    rectcontour_img = np.copy(img)
    for cnt in cnt_info_tuple_list:
        cv2.rectangle(rectcontour_img, (cnt[0],cnt[1]), (cnt[2],cnt[3]), (0,0,255), 10)

    return rectcontour_img

if __name__ == '__main__':
    file_name = "./data/3001-3200/IMG_4798.jpg"
    img = cv2.imread(file_name)

    threshold_img = get_thresholding_img(img)
    rectcontour_img = get_rectcontour_img(img, threshold_img)

    rectcontour_img = cv2.cvtColor(rectcontour_img, cv2.COLOR_BGR2RGB)
    plt.imshow(rectcontour_img)
    plt.show()
