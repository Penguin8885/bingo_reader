import sys
import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import bingo_reader as br


def show_thresholding_img(file_name):
    img = cv2.imread(file_name)

    threshold_img = br.get_hsv_thresholding_img(img, [30, 0, 100], [180, 100, 255])

    threshold_img = cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2RGB)
    plt.imshow(threshold_img)
    plt.show()

def show_frame_img(file_name, nc):
    img = cv2.imread(file_name)
    threshold_img = br.get_hsv_thresholding_img(img, [30, 0, 100], [180, 100, 255])

    #get rect contour
    _, contours, _ = cv2.findContours(threshold_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rect_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rect_contours.append([x, y, w, h])

    #noise cancellation
    if nc >= 2:
        rect_contours = br.noise_cancellation(rect_contours, err_ignore=True)

    #sort and configure matrix, optimize for bingo cards
    if nc >= 3:
        rect_contours = br.configure_matrix(rect_contours)
        rect_contours = br.optimize(rect_contours)
        rect_contours = [rect_contours[i][j] for i in range(5) for j in range(5)]
        rect_contours = [cnt for cnt in rect_contours if cnt is not None]

    #draw image
    frame_img = np.copy(img)
    for cnt in rect_contours:
        cv2.rectangle(frame_img, (cnt[0],cnt[1]), (cnt[0]+cnt[2],cnt[1]+cnt[3]), (0,0,255), 10)

        cv2.putText(
            frame_img,
            str(rect_contours.index(cnt)),
            (cnt[0],cnt[1]),
            cv2.FONT_HERSHEY_PLAIN,
            5, #font size
            (0,0,255),
            8, #line thickness
            cv2.LINE_AA
        )

    frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_img)
    plt.show()

def show_number_imgs(file_name):
    img = cv2.imread(file_name)

    threshold_img = br.get_hsv_thresholding_img(img, [30, 0, 100], [180, 100, 255])
    frame = br.get_frame(threshold_img)
    number_imgs = br.get_number_imgs(img, frame, [0, 0, 100], [255, 255, 255], post_blur_func=br.GaussianBlur)

    for i, number_img in enumerate(number_imgs):
        number_img = cv2.cvtColor(number_img, cv2.COLOR_GRAY2RGB)
        plt.imshow(number_img)
        plt.show()

def show_final_img(file_name):
    img = cv2.imread(file_name)

    threshold_img = br.get_hsv_thresholding_img(img, [30, 0, 100], [180, 100, 255])
    frame = br.get_frame(threshold_img)
    number_imgs = br.get_number_imgs(img, frame, [0, 0, 100], [255, 255, 255], post_blur_func=br.GaussianBlur)
    numbers = br.get_numbers(number_imgs)

    frame_img = np.copy(img)
    for i, cnt in enumerate(frame):
        ##draw frame
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

        ##draw number
        cv2.putText(
            frame_img,
            str(numbers[i]),
            (cnt[0],cnt[1]+65),
            cv2.FONT_HERSHEY_PLAIN,
            5, #font size
            (255,0,0),
            8, #line thickness
            cv2.LINE_AA
        )

    frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_img)
    plt.show()


if __name__ == '__main__':
    file_names = os.listdir("./error/")
    print(file_names)
    print("file number : ", end='')
    file_num = input()
    file_name = "./error/" + file_num + ".jpg"
    if file_num+".jpg" not in file_names and file_num+".JPG" not in file_names:
        print("input proper number")
        sys.exit(0)
    print("\n===========================")
    print("0 : threshold_img")
    print("1 : frame_img (noisy)")
    print("2 : frame_img (noise cancel)")
    print("3 : frame_img (optimize)")
    print("4 : number_imgs")
    print("5 : final_img")
    print("===========================")
    print("select type : ", end='')
    n = int(input())

    if n == 0:
        show_thresholding_img(file_name)
    elif n == 1:
        show_frame_img(file_name, nc=1)
    elif n == 2:
        show_frame_img(file_name, nc=2)
    elif n == 3:
        show_frame_img(file_name, nc=3)
    elif n == 4:
        show_number_imgs(file_name)
    elif n == 5:
        show_final_img(file_name)
