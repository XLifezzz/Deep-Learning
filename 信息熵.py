# import cv2
# import numpy as np
# import math
# tmp = []
# for i in range(256):
#     tmp.append(0)
# val = 0
# k = 0
# res = 0
# image = cv2.imread('C:/Users/admin/Desktop/20220333/1-1-959.jpg20220330.jpg',0)
# img = np.array(image)
# for i in range(len(img)):
#     for j in range(len(img[i])):
#         val = img[i][j]
#         tmp[val] = float(tmp[val] + 1)
#         k =  float(k + 1)
# for i in range(len(tmp)):
#     tmp[i] = float(tmp[i] / k)
# for i in range(len(tmp)):
#     if(tmp[i] == 0):
#         res = res
#     else:
#         res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
# print res
# import cv2
# import torch
# import os
# import collections
# from collections import Counter
# import math
#
# image_dir = 'C:/Users/admin/Desktop/20220333/1-1-959.jpg20220330.jpg'
# img = cv2.imread(image_dir, flags=cv2.IMREAD_GRAYSCALE)
# img = torch.from_numpy(img)
# compare_list = []
# for m in range(1, img.size()[0] - 1):
#     for n in range(1, img.size()[0] - 1):
#         sum_element = img[m - 1, n - 1] + img[m - 1, n] + img[m - 1, n + 1] + img[m, n - 1] + img[m, n + 1] + img[
#             m + 1, n - 1] + img[m + 1, n] + img[m + 1, n + 1]
#         sum_element = int(sum_element)
#         mean_element = sum_element // 8
#         pix = int(img[m, n])
#         temp = (pix, mean_element)
#         compare_list.append(temp)
#
# print(compare_list)
# compare_dict = collections.Counter(compare_list)
# H = 0.0
# for freq in compare_dict.values():
#     f_n2 = freq / img.size()[0] ** 2
#     log_f_n2 = math.log(f_n2)
#     h = -(f_n2 * log_f_n2)
#     H += h
#
# print(H)
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# import math
#
#
# def calc_array():
#     # img = cv2.imread('20201210_3.bmp',0)
#     # img = np.zeros([16,16]).astype(np.uint8)
#
#     a = [i for i in range(256)]
#     img = np.array(a).astype(np.uint8).reshape(16, 16)
#
#     hist_cv = cv2.calcHist([img], [0], None, [256], [0, 256])  # [0,256]的范围是0~255.返回值是每个灰度值出现的次数
#
#     # plt.subplot(111)
#     # plt.plot(hist_cv)
#     # plt.show()
#
#     P = hist_cv / (len(img) * len(img[0]))  # 概率
#     E = np.sum([p * np.log2(1 / p) for p in P])
#
#
# print(E)  # 熵
# calc_array()
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# import math
#
#
# def calc_2D_Entropy():
#     '''
#     邻域 3*3的小格子
#      __ __ __
#     |__|__|__|
#     |__||||__|
#     |__|__|__|
#     角点
#      __ __
#     ||||__|
#     |__|__|
#     边
#      __ __
#     |  |__|
#     ||||__|
#     |__|__|
#     '''
#     a = [i for i in range(256)]
#     img = np.array(a).astype(np.uint8).reshape(16, 16)
#
#     N = 1  # 设置邻域属性，目标点周围1个像素点设置为邻域，九宫格，如果为2就是25宫格...
#     S = img.shape
#     IJ = []
#     # 计算j
#     for row in range(S[0]):
#         for col in range(S[1]):
#             Left_x = np.max([0, col - N])
#             Right_x = np.min([S[1], col + N + 1])
#             up_y = np.max([0, row - N])
#             down_y = np.min([S[0], row + N + 1])
#             region = img[up_y:down_y, Left_x:Right_x]  # 九宫格区域
#             j = (np.sum(region) - img[row][col]) / ((2 * N + 1) ** 2 - 1)
#             IJ.append([img[row][col], j])
#     print(IJ)
#     # 计算F(i,j)
#     F = []
#     arr = [list(i) for i in set(tuple(j) for j in IJ)]  # 去重，会改变顺序，不过此处不影响
#     for i in range(len(arr)):
#         F.append(IJ.count(arr[i]))
#     print(F)
#     # 计算pij
#     P = np.array(F) / (img.shape[0] * img.shape[1])  # 也是img的W*H
#
#     # 计算熵
#
#     E = np.sum([p * np.log2(1 / p) for p in P])
#
#
# print(E)
#
# calc_2D_Entropy()
import math
import os

import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt


def dec2bin(p):
    floatbinstr = ""
    if p == 0:
        return floatbinstr

    for kk in range(len(str(p)) - 2):
        p *= 2
        if p > 1:
            floatbinstr += "1"
            p = p - int(p)
        else:
            floatbinstr += "0"

        if p == 0:
            break

        return str(floatbinstr)


def total_entropy(img):
    n = []
    P = []
    lenavg = []
    avg_sum = 0
    grey_lvl = 0
    k = 0
    res = 0
    # test = [[5,4,3,2,1]]
    weight = img.shape[0]
    height = img.shape[1]
    total = weight * height

    for i in range(256):
        n.append(0)

    for i in range(weight):
        for j in range(height):
            grey_lvl = img[i][j]
            n[grey_lvl] = float(n[grey_lvl] + 1)
            k = float(k + 1)

    for i in range(256):
        P.append(0)

    P = n
    for i in range(len(n)):
        P[i] = (n[i] / k)

    for i in range(256):
        lenavg.append(0)

    lenavg = P
    for i in range(len(n)):
        if P[i] == 0.0:
            continue
        lenavg[i] = lenavg[i] * len(dec2bin(lenavg[i]))
        avg_sum = lenavg[i] + avg_sum

    for i in range(len(n)):
        if (P[i] == 0):
            res = res
        else:
            res = float(res - P[i] * (math.log(P[i]) / math.log(2.0)))

    return res, avg_sum


if __name__ == '__main__':
    pic_dir = r"D:/OneDrive/OneDrive - zju.edu.cn/pictures - fuben -2"
    for filename in os.listdir(path=pic_dir):
        pic_path = os.path.join(pic_dir, filename)
        img_grey = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)
        H, lavg = total_entropy(img_grey)
        C = float(H / lavg * 100)
        R = 1 - (1 / C)
        print(H)
        # 将要输出保存的文件地址，若文件不存在，则会自动创建
        # fw = open("C:/Users/admin/Desktop/test.txt", 'w')
        #  这里平时print("test")换成下面这行，就可以输出到文本中了
        # fw.write("test")
        #  换行
        # fw.write("\n")

        # np.savetxt('test.txt', test, fmt='%d') 其中test.txt为要保存的文件filename，test为要保存的数组，fmt='%d'为数据保存格式，保存为整数



