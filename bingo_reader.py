import sys
import os
import shutil
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



def get_frame(binary_img, view_type=1):
    #get contour
    _, contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #get rect contour
    rect_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rect_contours.append([x, y, w, h])

    #noise cancellation
    rect_contours = noise_cancellation(rect_contours)

    #sort and configure matrix
    rect_contours = configure_matrix(rect_contours)

    #optimize for bingo cards
    rect_contours = optimize(rect_contours)

    #convert matrix to list
    rect_contours = [rect_contours[i][j] for i in range(5) for j in range(5)]
    rect_contours = [cnt for cnt in rect_contours if cnt is not None]

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

def center(cnt):
    return (cnt[0]+cnt[2]/2, cnt[1]+cnt[3]/2)

def configure_matrix(rect_contours):
    #clustering with x
    sorted_x = sorted(rect_contours, key=lambda x: center(x)[0])
    base_index = 0
    base_center = center(sorted_x[0])
    clusters_x = []
    for i, cnt in enumerate(sorted_x):
        if abs(base_center[0] - center(cnt)[0]) > 200:
            clusters_x.append(sorted_x[base_index:i])
            base_index = i
            base_center = center(cnt)
    clusters_x.append(sorted_x[base_index:])

    #clustering with y
    sorted_y = sorted(rect_contours, key=lambda x: center(x)[1])
    base_index = 0
    base_center = center(sorted_y[0])
    clusters_y = []
    for i, cnt in enumerate(sorted_y):
        if abs(base_center[1] - center(cnt)[1]) > 200:
            clusters_y.append(sorted_y[base_index:i])
            base_index = i
            base_center = center(cnt)
    clusters_y.append(sorted_y[base_index:])

    #make matrix
    rc_mat = [[None for i in range(5)] for j in range(5)]
    for cnt in rect_contours:
        for i, x_line in enumerate(clusters_x):
            for j, y_line in enumerate(clusters_y):
                if cnt in x_line and cnt in y_line:
                    rc_mat[i][j] = cnt
                    break
            else:
                continue
            break

    return rc_mat

def noise_cancellation(rect_contours, err_ignore=False):
    passer = []
    for cnt in rect_contours:
        area = cnt[2]*cnt[3]
        ratio = cnt[2]/cnt[3]
        if (area > 70000 and area < 200000) and (ratio > 0.7 and ratio < 1.3):
            passer.append(cnt)

    if len(passer) > 25:
        passer_copy = passer[:]
        for cnt1 in passer[:]:
            for cnt2 in passer[:]:
                if cnt1 == cnt2:
                    continue
                top_l1 = (cnt1[0], cnt1[1])
                btm_r1 = (cnt1[0]+cnt1[2], cnt1[1]+cnt1[3])
                top_l2 = (cnt2[0], cnt2[1])
                btm_r2 = (cnt2[0]+cnt2[2], cnt2[1]+cnt2[3])
                if top_l1[0] <= top_l2[0] and top_l2[0] <= btm_r1[0] \
                 and top_l1[1] <= top_l2[1] and top_l2[1] <= btm_r1[1] \
                 and top_l1[0] <= btm_r2[0] and btm_r2[0] <= btm_r1[0] \
                 and top_l1[1] <= btm_r2[1] and btm_r2[1] <= btm_r1[1]:
                    passer.remove(cnt2)


    if err_ignore is False and len(passer) > 25:
        raise Exception("Noise Cancellation Filure, the number of cnt is over 25")

    return passer

def optimize(rect_contours0):
    #weak-point: if there are the number of inappropriate coordinates more than 2 in a line, this could not work well

    #copy
    rect_contours = rect_contours0.copy()

    #convert None -> tmp_cnt
    for i in range(5):
        for j in range(5):
            if rect_contours[i][j] is None:
                rect_contours[i][j] = [5, 5, 1, 10, "add"] #with label

    #optimize about x
    col_flag = []
    rc = rect_contours.copy()
    for i, col in enumerate(rc):
        passer = [cnt for cnt in col if cnt[2]/cnt[3] > 0.95 and cnt[2]/cnt[3] < 1.05]
        not_passer = [cnt for cnt in col if cnt[2]/cnt[3] <= 0.95 or cnt[2]/cnt[3] >= 1.05]

        if len(not_passer) == 0:
            col_flag.append(True)
            continue
        if len(passer) == 0:
            col_flag.append(False)
            continue
        else:
            col_flag.append(True)

        #calculate median
        sorted_ = sorted(passer, key=lambda x: x[0])
        median_l = sorted_[int(len(sorted_)/2-1)][0]
        sorted_ = sorted(passer, key=lambda x: x[0]+x[2])
        median_r = sorted_[int(len(sorted_)/2-0.5)]
        median_r = median_r[0] + median_r[2]

        #correct
        for cnt in not_passer:
            j = rc[i].index(cnt)
            if abs(cnt[0]-median_l) > 30:
                cnt[2] += cnt[0] - median_l
                cnt[0] = median_l
                if len(cnt) == 4:
                    cnt.append("correct") #label
            if abs(cnt[0]+cnt[2]-median_r) > 30:
                cnt[2] += median_r - (cnt[0] + cnt[2])
                if len(cnt) == 4:
                    cnt.append("correct") #label
            rect_contours[i][j] = cnt

    #optimize about y
    row_flag = []
    row1, row2, row3, row4, row5 = zip(*rect_contours)
    rc = [row1, row2, row3, row4, row5]
    for i, row in enumerate(rc):
        passer = [cnt for cnt in row if cnt[2]/cnt[3] > 0.95 and cnt[2]/cnt[3] < 1.05]
        not_passer = [cnt for cnt in row if cnt[2]/cnt[3] <= 0.95 or cnt[2]/cnt[3] >= 1.05]

        if len(not_passer) == 0:
            row_flag.append(True)
            continue
        if len(passer) == 0:
            row_flag.append(False)
            continue
        else:
            row_flag.append(True)

        #calculate median
        sorted_ = sorted(passer, key=lambda x: x[1])
        median_t = sorted_[int(len(sorted_)/2-1)][1]
        sorted_ = sorted(passer, key=lambda x: x[1]+x[3])
        median_b = sorted_[int(len(sorted_)/2-0.5)]
        median_b = median_b[1] + median_b[3]

        #correct
        for cnt in not_passer:
            j = rc[i].index(cnt)
            if abs(cnt[1]-median_t) > 30:
                cnt[3] += cnt[1] - median_t
                cnt[1] = median_t
                if len(cnt) == 4:
                    cnt.append("correct") #label
            if abs(cnt[1]+cnt[3]-median_b) > 30:
                cnt[3] += median_b - (cnt[1] + cnt[3])
                if len(cnt) == 4:
                    cnt.append("correct") #label
            rect_contours[j][i] = cnt

    #optimize again
    if False in col_flag:
        ##serach TRUE column and get section_size
        index1 = None; index2 = None
        for i, flag in enumerate(col_flag):
            if flag is True:
                if index1 is None:
                    index1 = i
                    continue
                else:
                    index2 = i
                    break
        if index2 is None:
            raise Exception("Correction Failure")
        section_size = int((rect_contours[index2][0][0] - rect_contours[index1][0][0]) / (index2 - index1))

        ##correction
        for i in range(4):
            if col_flag[i] is True and col_flag[i+1] is False:
                for j in range(5):
                    rect_contours[i+1][j][0] = rect_contours[i][j][0] + section_size
                    rect_contours[i+1][j][2] = rect_contours[i][j][2]
                    if len(rect_contours[i+1][j]) == 4:
                        rect_contours[i+1][j].append("correct") #label
        for i in range(4,0,-1):
            if col_flag[i] is True and col_flag[i-1] is False:
                for j in range(5):
                    rect_contours[i-1][j][0] = rect_contours[i][j][0] - section_size
                    rect_contours[i-1][j][2] = rect_contours[i][j][2]
                    if len(rect_contours[i-1][j]) == 4:
                        rect_contours[i-1][j].append("correct") #label
    if False in row_flag:
        ##serach TRUE row and get section_size
        index1 = None; index2 = None
        for i, flag in enumerate(row_flag):
            if flag is True:
                if index1 is None:
                    index1 = i
                    continue
                else:
                    index2 = i
                    break
        if index2 is None:
            raise Exception("Correction Failure")
        section_size = int((rect_contours[0][index2][1] - rect_contours[0][index1][1]) / (index2 - index1))

        ##correction
        for i in range(4):
            if row_flag[i] is True and row_flag[i+1] is False:
                for j in range(5):
                    rect_contours[j][i+1][1] = rect_contours[j][i][1] + section_size
                    rect_contours[j][i+1][3] = rect_contours[j][i][3]
                    if len(rect_contours[j][i-1]) == 4:
                        rect_contours[j][i-1].append("correct") #label
        for i in range(4,0,-1):
            if row_flag[i] is True and row_flag[i-1] is False:
                for j in range(5):
                    rect_contours[j][i-1][1] = rect_contours[j][i][1] - section_size
                    rect_contours[j][i-1][4] = rect_contours[j][i][4]
                    if len(rect_contours[j][i-1]) == 4:
                        rect_contours[j][i-1].append("correct") #label


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
    img = np.load("./num_img/figureX_40.npy")
    img = (img<127).astype(np.int)
    img[img==0] = -1
    base_40_img = img

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
            if sum(sum(base_40_img*c))/10000 > 90:
                number = 40
            else:
                raise Exception("cannnot read number")

        numbers.append(number)

    print(numbers)

    #check duplication
    if len(np.unique(numbers)) < 25:
        raise Exception("Duplication is detected")

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
    with open("bingo_numbers.csv", "w") as csv_f:
        writer = csv.writer(csv_f, lineterminator='\n')
        for file_name in file_names:
            if file_name == ".gitkeep":
                continue
            try:
                print("\n", file_name, "\a")
                img = cv2.imread("./data/"+file_name)
                threshold_img = get_hsv_thresholding_img(img, [30, 0, 100], [180, 100, 255])
                frame = get_frame(threshold_img)
                number_imgs = get_number_imgs(img, frame, [0, 0, 100], [255, 255, 255], post_blur_func=GaussianBlur)
                numbers = get_numbers(number_imgs)
                write_img("./result/"+file_name, img, frame, numbers)

                #output numbers to csv
                root, ext = os.path.splitext(file_name)
                numbers.insert(0, root)
                writer.writerow(numbers)
            except Exception as e:
                print(e, "\n", "pass "+file_name)
                writer.writerow(["###", e.args[0]])
                shutil.copyfile("./data/"+file_name, "./error/"+file_name)
                os.remove("./data/"+file_name)
            finally:
                pass
