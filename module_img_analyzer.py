import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def get_gray_thresholding_img(bgr_img, lower, upper):
    blur_img = cv2.GaussianBlur(bgr_img, (15,15), 0)

    gray_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)
    _, threshold_img = cv2.threshold(gray_img, lower, upper, cv2.THRESH_BINARY)

    return threshold_img


def get_bgr_thresholdimg_img(bgr_img, lower, upper):
    blur_img = cv2.GaussianBlur(bgr_img, (15,15), 0)

    threshold_img = cv2.inRange(
        bgr_img,
        np.array(lower, np.uint8),
        np.array(upper, np.uint8)
    )

    threshold_img = cv2.bitwise_not(threshold_img)

    return threshold_img


def get_hsv_thresholding_img(bgr_img, lower, upper):
    blur_img = cv2.GaussianBlur(bgr_img, (15,15), 0)

    hsv_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2HSV)
    threshold_img = cv2.inRange(
        hsv_img,
        np.array(lower, np.uint8),
        np.array(upper, np.uint8)
    )

    threshold_img = cv2.bitwise_not(threshold_img)

    return threshold_img


def get_rect_contour_img(bgr_img, binary_img):
    #get contour
    _, contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #get rect contour
    noisy_rect_contour_img = np.copy(bgr_img)
    passer = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(noisy_rect_contour_img, (x,y), (x+w,y+h), (0,0,255), 10)

        #noise cancellation
        area = w*h
        ratio = w/h
        if (area > 70000 and area < 200000) and (ratio > 0.7 and ratio < 1.3):
            passer.append((x, y, w, h))
    rect_contours = passer

    #########
    if len(rect_contours) != 25:
        print("Warning: The number of contours is not 25")
    #########

    #sort (clustering)
    rect_contours.sort()

    base = rect_contours[0]
    cluster = []
    rect_contour_clusters = []
    for cnt in rect_contours:
        if abs(base[0] - cnt[0]) < 100:
            cluster.append(cnt)
        else:
            rect_contour_clusters.append(cluster)
            base = cnt
            cluster = [cnt]
    rect_contour_clusters.append(cluster)

    for cluster in rect_contour_clusters:
        cluster.sort(key=lambda x: x[1])

    rect_contours = []
    for cluster in rect_contour_clusters:
        for cnt in cluster:
            rect_contours.append(cnt)

    print("# \t (x, y, area, ratio)")
    for cnt in rect_contours:
        print(rect_contours.index(cnt), "\t", (cnt[0],cnt[1],cnt[2]*cnt[3],cnt[2]/cnt[3]))

    #draw image
    rect_contour_img = np.copy(bgr_img)
    for cnt in rect_contours:
        cv2.rectangle(rect_contour_img, (cnt[0],cnt[1]), (cnt[0]+cnt[2],cnt[1]+cnt[3]), (0,0,255), 10)

        cv2.putText(
            rect_contour_img,
            str(rect_contours.index(cnt)),
            (cnt[0],cnt[1]),
            cv2.FONT_HERSHEY_PLAIN,
            5, #font size
            (0,0,255),
            8, #line thickness
            cv2.LINE_AA
        )

    return noisy_rect_contour_img, rect_contour_img


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


def _show_thresholding_img(file_name):
    img = cv2.imread(file_name)

    threshold_img = get_gray_thresholding_img(img, 80, 255)
    #threshold_img = get_bgr_thresholdimg_img(img, [200, 200, 200], [255, 255, 255])
    #threshold_img = get_hsv_thresholding_img(img, [0, 0, 0], [255, 200, 255])

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
