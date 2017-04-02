import sys
import os
import csv
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def GaussianBlur(img, size=15):
    return cv2.GaussianBlur(img, (size, size), 0)
def medianBlur(img, size=3):
    return cv2.medianBlur(img, size)



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



def get_frame(bgr_img, binary_img, nc=True, view_type=1):
    #get contour
    _, contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #get rect contour
    rect_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rect_contours.append([x, y, w, h])

    if nc is True or nc == 0:
        #noise cancellation
        rect_contours = noise_cancellation(rect_contours)
    if nc is True or nc != 0:
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
    if view_type == 1:
        for cnt in rect_contours:
            if len(cnt) >= 5:
                print(
                    rect_contours.index(cnt), "\t",
                    (cnt[0],cnt[1],cnt[0]+cnt[2],cnt[1]+cnt[3]),
                    (cnt[2]*cnt[3],"{:.4f}".format(cnt[2]/cnt[3])), "  ",
                    "--"+str(cnt[4])
                )
    elif view_type == 2:
        print("# \t (x0, y0, x1, y1) (area, ratio) --modification")
        for cnt in rect_contours:
            print(
                rect_contours.index(cnt), "\t",
                (cnt[0],cnt[1],cnt[0]+cnt[2],cnt[1]+cnt[3]),
                (cnt[2]*cnt[3],"{:.4f}".format(cnt[2]/cnt[3])),
                "--"+str(cnt[4]) if len(cnt) == 5 else ""
            )
    elif view_type == 0:
        pass
    else:
        pass
    ########

    return rect_contours

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

    #array of the median
    med_l = []
    med_r = []
    med_t = []
    med_b = []

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
        if len(serted_) > 0:
            median_l = sorted_[int(len(sorted_)/2-1)][0]
        else:
            median_l = cluster[0][0]
        ##search right median
        sorted_ = sorted(cluster, key=lambda x: x[0]+x[2])
        for cnt in sorted_[:]:
            ratio = cnt[2] / cnt[3]
            if ratio < 0.95 or ratio > 1.05:
                sorted_.remove(cnt)
        if len(serted_) > 0:
            median_r = sorted_[int(len(sorted_)/2-0.5)]
            median_r = median_r[0] + median_r[2]
        else:
            median_r = cluster[0][0] + cluster[0][2]

        med_l.append(median_l)
        med_r.append(median_r)

        ##modify
        for cnt in cluster:
            ratio = cnt[2] / cnt[3]
            if ratio < 0.95 or ratio > 1.05:
                if abs(cnt[0]-median_l) > 30:
                    cnt[2] += cnt[0] - median_l
                    cnt[0] = median_l
                    cnt.append("correct")
                if abs(cnt[0]+cnt[2]-median_r) > 30:
                    cnt[2] += median_r - (cnt[0] + cnt[2])
                    if len(cnt) == 4:
                        cnt.append("correct")
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
        ##search top median
        sorted_ = sorted(cluster, key=lambda x: x[1])
        for cnt in sorted_[:]:
            ratio = cnt[2] / cnt[3]
            if ratio < 0.95 or ratio > 1.05:
                sorted_.remove(cnt)
        if len(serted_) > 0:
            median_t = sorted_[int(len(sorted_)/2-1)][1]
        else:
            median_t = cluster[0][1]
        ##search bottom median
        sorted_ = sorted(cluster, key=lambda x: x[1]+x[3])
        for cnt in sorted_[:]:
            ratio = cnt[2] / cnt[3]
            if ratio < 0.95 or ratio > 1.05:
                sorted_.remove(cnt)
        if len(serted_) > 0:
            median_b = sorted_[int(len(sorted_)/2-0.5)]
            median_b = median_b[1] + median_b[3]
        else:
            median_b = cluster[0][1] + cluster[0][3]

        med_t.append(median_t)
        med_b.append(median_b)

        ##modify
        for cnt in cluster:
            ratio = cnt[2] / cnt[3]
            if ratio < 0.95 or ratio > 1.05:
                if abs(cnt[1]-median_t) > 30:
                    cnt[3] += cnt[1] - median_t
                    cnt[1] = median_t
                    cnt.append("correct")
                if abs(cnt[1]+cnt[3]-median_b) > 30:
                    cnt[3] += median_b - (cnt[1] + cnt[3])
                    if len(cnt) == 4:
                        cnt.append("correct")

    #to gather into one array
    rect_contours = []
    for cluster in rect_contour_clusters:
        for cnt in cluster:
            rect_contours.append(cnt)


    ######## <step 3> fill the blank ########

    #search the blank and fill it
    med_l.sort()
    med_r.sort()
    med_t.sort()
    med_b.sort()
    for i in range(5):
        for j in range(5):
            flag = False
            for cnt in rect_contours[:]:
                if abs(cnt[0]-med_l[i]) < 100 and abs(cnt[1]-med_t[j]) < 100:
                    flag = True
                    break
            if flag is False:
                rect_contours.append([med_l[i],med_t[j],(med_r[i]-med_l[i]),(med_b[j]-med_t[j]), "add"])

    if len(rect_contours) != 25:
        print("error : the number of cntours is not 25")
        sys.exit(0)

    return rect_contours



def get_number_imgs(img, frame, lower, upper, pre_blur_func=GaussianBlur, post_blur_func=None, blur_size=5):
    number_imgs = []
    for cnt in frame:
        rect = img[cnt[1]:(cnt[1]+cnt[3]), cnt[0]:(cnt[0]+cnt[2])]
        rect = cv2.resize(rect, (1000,1000))

        threshold_img = get_hsv_thresholding_img(
            rect,
            lower,
            upper,
            pre_blur_func=pre_blur_func,
            post_blur_func=post_blur_func,
            blur_size=blur_size
        )
        threshold_img = cv2.bitwise_not(threshold_img)

        number_imgs.append(threshold_img)
    return number_imgs

def get_numbers(number_imgs):
    #load base number figures
    base_num_img = []
    for i in range(10):
        img = np.load("./num_img/figure_"+str(i)+".npy")
        img = (img<127).astype(np.int)
        img[img==0] = -1
        base_num_img.append(img)

    #convert number_imgs to numbers
    numbers = []
    for i in range(25):
        #skip free zone
        if i == 12:
            numbers.append(0)
            continue

        number_img = number_imgs[i]
        #get contour
        cp = np.array(number_img)
        _, contours, _ = cv2.findContours(number_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #get rect contour & number charactor images
        char_imgs = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w*h > 900000 or w*h < 50000:
                continue #filter
            char_img = cv2.resize(cp[y:(y+h), x:(x+w)], (1000,1000))
            char_imgs.append([x, char_img]) #get char_img
        char_imgs.sort(key=lambda x:x[0])

        #calculate correlation & recognize number
        number = 0
        for char_img in char_imgs:
            char_img = char_img[1]

            #get the correlation & the number
            c = (char_img<127).astype(np.int)
            c[c==0] = -1
            correlation = [int(sum(sum(base_num_img[i]*c))/10000) for i in range(10)]
            for i in range(10):
                if correlation[i] < 70:
                    correlation[i] = 0 #filter
            n = correlation.index(max(correlation))

            #exception check
            if sum(correlation) > 100 and max(correlation) < 90:
                plt.imshow(cv2.cvtColor(char_img, cv2.COLOR_GRAY2RGB))
                plt.show()

            #result
            number = number*10 + n

        #append results
        if number == 0:
            print("warning: can not read number, guess the number is 40")
            number = 40
        numbers.append(number)

    print(numbers)

    #check duplication
    if len(np.unique(numbers)) < 25:
        print("error : Duplication is detected")
        sys.exit(0)

    return numbers



def write_img(file_name, img, frame, numbers):
    #draw image
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

    #save
    cv2.imwrite(file_name, frame_img)



if __name__ == '__main__':
    file_names = os.listdir("./data/")
    with open('bingo_numbers.csv', 'w') as f:
        for file_name in file_names:
            print("\n", file_name, "\a")
            img = cv2.imread("./data/"+file_name)
            threshold_img = get_hsv_thresholding_img(img, [30, 0, 100], [180, 100, 255])
            frame = get_frame(img, threshold_img)
            number_imgs = get_number_imgs(img, frame, [0, 0, 100], [255, 255, 255], post_blur_func=GaussianBlur)
            numbers = get_numbers(number_imgs)
            write_img("./result/"+file_name, img, frame, numbers)

            #output numbers to csv
            root, ext = os.path.splitext(file_name)
            numbers.insert(0, root)
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(numbers)
