import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file_name = "IMG_4798.jpg"

    img = cv2.imread("./data/3001-3200/" + file_name)

    blur_img = cv2.GaussianBlur(img, (15,15), 0)

    hsv_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2HSV)
    threshold_img = cv2.inRange(
        hsv_img,
        np.array([30,0,100], np.uint8),
        np.array([180,100,255], np.uint8)
    )
    threshold_img = cv2.bitwise_not(threshold_img)

    _, contours, _ = cv2.findContours(threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cntInfoTuple_list = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cntInfoTuple_list.append((x, y, x+w, y+h))

    img_with_contours = np.copy(img)
    for cnt in cntInfoTuple_list:
        cv2.rectangle(img_with_contours, (cnt[0],cnt[1]), (cnt[2],cnt[3]), (0,0,255), 10)

    img_with_contours = cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB)
    plt.imshow(img_with_contours)
    plt.show()
