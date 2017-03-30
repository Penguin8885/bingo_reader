import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import bingo_reader as br


def show_thresholding_img(file_name):
    img = cv2.imread(file_name)

    threshold_img = br.get_hsv_thresholding_img(img, [30, 0, 100], [180, 100, 255])

    cv2.imwrite("thr_sample.jpg", threshold_img)
    threshold_img = cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2RGB)
    plt.imshow(threshold_img)
    plt.show()

def show_frame_img(file_name, nc=True):
    img = cv2.imread(file_name)

    threshold_img = br.get_hsv_thresholding_img(img, [30, 0, 100], [180, 100, 255])
    frame = br.get_frame(img, threshold_img, nc=nc, view_type=2)

    #draw image
    frame_img = np.copy(img)
    for cnt in frame:
        cv2.rectangle(frame_img, (cnt[0],cnt[1]), (cnt[0]+cnt[2],cnt[1]+cnt[3]), (0,0,255), 10)

        cv2.putText(
            frame_img,
            str(frame.index(cnt)),
            (cnt[0],cnt[1]),
            cv2.FONT_HERSHEY_PLAIN,
            5, #font size
            (0,0,255),
            8, #line thickness
            cv2.LINE_AA
        )

    cv2.imwrite("cnt_sample.jpg", frame_img)
    frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_img)
    plt.show()

def show_number_imgs(file_name):
    img = cv2.imread(file_name)

    threshold_img = br.get_hsv_thresholding_img(img, [30, 0, 100], [180, 100, 255])
    frame = br.get_frame(img, threshold_img)
    number_imgs = br.get_number_imgs(img, frame, [0, 0, 100], [255, 255, 255], post_blur_func=br.GaussianBlur)

    for number_img in number_imgs:
        cv2.imwrite("number"+str(i)+"_sample.jpg", number_img)
        number_img = cv2.cvtColor(number_img, cv2.COLOR_GRAY2RGB)
        plt.imshow(number_img)
        plt.show()


if __name__ == '__main__':
    #4879
    #4874,4877,4881
    file_name = "./data/IMG_4877.jpg"

    #show_thresholding_img(file_name)
    #show_frame_img(file_name, nc=True)
    show_number_imgs(file_name)
