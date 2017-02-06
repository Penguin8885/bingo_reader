import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def get_gray_thresholding_img(img):
    return False

def get_hsv_thresholding_img(img):
    blur_img = cv2.GaussianBlur(img, (15,15), 0)

    hsv_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2HSV)
    threshold_img = cv2.inRange(
        hsv_img,
        np.array([30,0,100], np.uint8),
        np.array([180,100,255], np.uint8)
    )

    threshold_img = cv2.bitwise_not(threshold_img)

    return threshold_img


def get_rect_contour_img(img, binary_img):
    _, contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    noisy_rect_contour_img = np.copy(img)
    passer = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(noisy_rect_contour_img, (x,y), (x+w,y+h), (0,0,255), 10)

        raito = w/h
        area = w*h
        if (ratio > 0.7 and ratio < 1.3) and (area > 70000 and area < 200000):
            passer.append((x, y, x+w, y+h))
    rect_cntours = passer

    rect_conturs.sort(key=lambda tuple_: tuple_[2]*tuple_[3], reverse=True)
    for cnt in rect_conturs:
        print(rect_conturs.index(cnt)+1, "\t", cnt[2]*cnt[3])

    rect_contour_img = np.copy(img)
    for cnt in rect_conturs:
        cv2.rectangle(rect_contour_img, (cnt[0],cnt[1]), (cnt[0]+cnt[2],cnt[1]+cnt[3]), (0,0,255), 10)

    return noisy_rect_contour_img, rect_contour_img


def get_circle_contour_img(img, binary_img):
    circles = cv2.HoughCircles(binary_img, cv2.HOUGH_GRADIENT, dp=5, minDist=400, minRadius=180, maxRadius=200)

    circle_contour_img = np.copy(img)
    if circles is not None and len(circles) > 0:
        for (x, y, r) in circles[0]:
            print((x,y,r))
            x = int(x); y = int(y); r = int(r)
            cv2.circle(circle_contour_img, (x,y), r, (0,255,0), 10)
            cv2.rectangle(circle_contour_img, (int(x-r),int(y-r)), (int(x+r),int(y+r)), (0,0,255), 10)
    else:
        print("cannot detect circles")

    return circle_contour_img


def show_contour_img(file_name):
    img = cv2.imread(file_name)

    threshold_img = get_hsv_thresholding_img(img)
    _, contour_img = get_rect_contour_img(img, threshold_img)
    #contour_img = get_circle_contour_img(img, threshold_img)

    contour_img = cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)
    plt.imshow(contour_img)
    plt.show()


def write_imgs(file_name, output_name):
    img = cv2.imread(file_name)

    threshold_img = get_hsv_thresholding_img(img)
    noisy_rect_contour_img, rect_contour_img = get_rect_contour_img(img, threshold_img)

    cv2.imwrite("0_" + output_name, threshold_img)
    cv2.imwrite("1_" + output_name, noisy_rect_contour_img)
    cv2.imwrite("2_" + output_name, rect_contour_img)


if __name__ == '__main__':
    show_contour_img("./data/3001-3200/IMG_4798.jpg")
    #write_imgs("./data/3001-3200/IMG_4798.jpg", "sample.jpg")
