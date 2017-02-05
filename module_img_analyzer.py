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
    noisy_rectcontour_img = np.copy(img)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cnt_info_tuple_list.append((x, y, w, h))
        cv2.rectangle(noisy_rectcontour_img, (x,y), (x+w,y+h), (0,0,255), 10)

    passer = []
    for cnt in cnt_info_tuple_list:
        ratio = cnt[2] / cnt[3]
        area = cnt[2] * cnt[3]
        if (ratio > 0.7 and ratio < 1.3) and (area > 70000 and area < 200000):
            passer.append((cnt))
    cnt_info_tuple_list = passer

    cnt_info_tuple_list.sort(key=lambda tuple_: tuple_[2]*tuple_[3], reverse=True)
    for cnt in cnt_info_tuple_list:
        print(cnt_info_tuple_list.index(cnt)+1, "\t", cnt[2]*cnt[3])

    rectcontour_img = np.copy(img)
    for cnt in cnt_info_tuple_list:
        cv2.rectangle(rectcontour_img, (cnt[0],cnt[1]), (cnt[0]+cnt[2],cnt[1]+cnt[3]), (0,0,255), 10)

    return noisy_rectcontour_img, rectcontour_img

def show_rectcontour_img(file_name):
    img = cv2.imread(file_name)

    threshold_img = get_thresholding_img(img)
    _, rectcontour_img = get_rectcontour_img(img, threshold_img)

    rectcontour_img = cv2.cvtColor(rectcontour_img, cv2.COLOR_BGR2RGB)
    plt.imshow(rectcontour_img)
    plt.show()

def write_imgs(file_name, output_name):
    img = cv2.imread(file_name)

    threshold_img = get_thresholding_img(img)
    noisy_rectcontour_img, rectcontour_img = get_rectcontour_img(img, threshold_img)

    cv2.imwrite("0_" + output_name, threshold_img)
    cv2.imwrite("1_" + output_name, noisy_rectcontour_img)
    cv2.imwrite("2_" + output_name, rectcontour_img)


if __name__ == '__main__':
    show_rectcontour_img("./data/3001-3200/IMG_4798.jpg")
    write_imgs("./data/3001-3200/IMG_4798.jpg", "sample.jpg")
