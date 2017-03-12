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
    #get contour
    _, contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #get rect contour
    noisy_rect_contour_img = np.copy(bgr_img)
    rect_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(noisy_rect_contour_img, (x,y), (x+w,y+h), (0,0,255), 10)
        rect_contours.append([x, y, w, h])

    #noise cancellation
    rect_contours = noise_cancellation(rect_contours)

    #optimize for bingo cards
    rect_contours = optimize(rect_contours)

    #sort
    ##sort with x-axis
    rect_contours.sort(key=lambda x: x[0])

    ##clustering with x-axis
    base = rect_contours[0]
    cluster = []
    rect_contour_clusters = []
    for cnt in rect_contours:
        if abs(base[0] - cnt[0]) < 200:
            cluster.append(cnt)
        else:
            rect_contour_clusters.append(cluster)
            base = cnt
            cluster = [cnt]
    rect_contour_clusters.append(cluster)

    ##sort with y-axis
    for cluster in rect_contour_clusters:
        cluster.sort(key=lambda x: x[1])

    ##to gather into one array
    rect_contours = []
    for cluster in rect_contour_clusters:
        for cnt in cluster:
            rect_contours.append(cnt)

    ########
    print("# \t (x, y, area, ratio)")
    for cnt in rect_contours:
        print(rect_contours.index(cnt), "\t", (cnt[0],cnt[1],cnt[2]*cnt[3],cnt[2]/cnt[3]))
    ########

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

def noise_cancellation(rect_contours):
    passer = []
    for cnt in rect_contours:
        area = cnt[2]*cnt[3]
        ratio = cnt[2]/cnt[3]
        if (area > 70000 and area < 200000) and (ratio > 0.7 and ratio < 1.3):
            passer.append(cnt)
    return passer

def optimize(rect_contours0):
    #weak-point: if there are the number of inappropriate coordinates more than 2 in a line, this could not work well

    #copy
    rect_contours = rect_contours0.copy()


    ######## <step 1> Correction about x-axis ########

    #clustering with x-axis
    ##sort with x-axis
    rect_contours.sort(key=lambda x: x[0])
    ##clustering with x-axis
    base = rect_contours[0]
    cluster = []
    rect_contour_clusters = []
    for cnt in rect_contours:
        if abs(base[0] - cnt[0]) < 200:
            cluster.append(cnt)
        else:
            rect_contour_clusters.append(cluster)
            base = cnt
            cluster = [cnt]
    rect_contour_clusters.append(cluster)

    #optimize about x-axis
    for cluster in rect_contour_clusters:
        ##search left medial
        sorted_ = sorted(cluster, key=lambda x: x[0])
        for cnt in sorted_[:]:
            ratio = cnt[2] / cnt[3]
            if ratio < 0.95 or ratio > 1.05:
                sorted_.remove(cnt)
        median_l = sorted_[int(len(sorted_)/2)][0]
        ##search right median
        sorted_ = sorted(cluster, key=lambda x: x[0]+x[2])
        for cnt in sorted_[:]:
            ratio = cnt[2] / cnt[3]
            if ratio < 0.95 or ratio > 1.05:
                sorted_.remove(cnt)
        median_r = sorted_[int(len(sorted_)/2+0.5)]
        median_r = median_r[0] + median_r[2]

        ##modify
        for cnt in cluster:
            ratio = cnt[2] / cnt[3]
            if ratio < 0.95 or ratio > 1.05:
                if abs(cnt[0]-median_l) > 30:
                    cnt[2] += cnt[0] - median_l
                    cnt[0] = median_l
                if abs(cnt[0]+cnt[2]-median_r) > 30:
                    cnt[2] += median_r - (cnt[0] + cnt[2])

    #to gather into one array
    rect_contours = []
    for cluster in rect_contour_clusters:
        for cnt in cluster:
            rect_contours.append(cnt)


    ######## <step 2> Correction about y-axis ########

    #clustering with y-axis
    ##sort with y-axis
    rect_contours.sort(key=lambda x: x[1])
    ##clustering with y-axis
    base = rect_contours[0]
    cluster = []
    rect_contour_clusters = []
    for cnt in rect_contours:
        if abs(base[1] - cnt[1]) < 200:
            cluster.append(cnt)
        else:
            rect_contour_clusters.append(cluster)
            base = cnt
            cluster = [cnt]
    rect_contour_clusters.append(cluster)

    #optimize about y-axis
    for cluster in rect_contour_clusters:
        ##search top medial
        sorted_ = sorted(cluster, key=lambda x: x[1])
        for cnt in sorted_[:]:
            ratio = cnt[2] / cnt[3]
            if ratio < 0.95 or ratio > 1.05:
                sorted_.remove(cnt)
        median_t = sorted_[int(len(sorted_)/2)][1]
        ##search bottom median
        sorted_ = sorted(cluster, key=lambda x: x[1]+x[3])
        for cnt in sorted_[:]:
            ratio = cnt[2] / cnt[3]
            if ratio < 0.95 or ratio > 1.05:
                sorted_.remove(cnt)
        median_b = sorted_[int(len(sorted_)/2+0.5)]
        median_b = median_b[1] + median_b[3]

        ##modify
        for cnt in cluster:
            ratio = cnt[2] / cnt[3]
            if ratio < 0.95 or ratio > 1.05:
                if abs(cnt[1]-median_t) > 30:
                    cnt[3] += cnt[1] - median_t
                    cnt[1] = median_t
                if abs(cnt[1]+cnt[3]-median_b) > 30:
                    cnt[3] += median_b - (cnt[1] + cnt[3])

    #to gather into one array
    rect_contours = []
    for cluster in rect_contour_clusters:
        for cnt in cluster:
            rect_contours.append(cnt)


    ######## <step 3> fill the blank ########


    return rect_contours
