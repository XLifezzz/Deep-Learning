import cv2
import numpy as np
import math
from os import listdir
from os.path import isfile, join
import re
import matplotlib.pyplot as plt
import time
import datetime
import glob
from imutils import contours
import argparse
import myutils
import os
import shutil

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#清空文件夹里面的内容
def clear_folder(folder_path):
    if os.path.isdir(folder_path):
        # 如果文件夹存在，则删除里面的所有文件和子文件夹
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('删除文件或文件夹时出现错误：', e)
    else:
        print('指定的路径不存在或不是一个文件夹')


#通道划分+预处理
def pre(rgbpath, leftpath, rightpath):
    imagefolderpath = rgbpath
    leftpath = leftpath
    rightpath = rightpath
    onlyfiles1 = [f for f in listdir(imagefolderpath) if isfile(join(imagefolderpath, f))]
    onlyfiles1.sort(key=lambda f: int(re.sub('\D', '', f)))
    imagepathlist1 = onlyfiles1
    pinum = len(imagepathlist1)
    for i in range(pinum):
        #print(i)
        image0 = plt.imread(join(imagefolderpath, imagepathlist1[i]))
        image0 = cv2.pyrDown(image0)
        #左通道划分
        image_left_path = join(leftpath, imagepathlist1[i])
        # image_left = image0[0:390, 0:285]
        image_left = image0[0:195, 0:143]
        #cv2.imwrite(image_left_path, image_left)
        #右通道划分
        image_right = image0[0:195, 172:315]
        image_right_path = join(rightpath, imagepathlist1[i])
        #cv2.imwrite(image_right_path, image_right)
        #预处理-高斯滤波
        aussian_left = cv2.GaussianBlur(image_left, (7, 7), 1)
        aussian_right = cv2.GaussianBlur(image_right, (7, 7), 1)
        #预处理-二值化转换
        r1, g1, b1 = aussian_left[:, :, 0], aussian_left[:, :, 1], aussian_left[:, :, 2]
        r2, g2, b2 = aussian_right[:, :, 0], aussian_right[:, :, 1], aussian_right[:, :, 2]
        img1 = -0.2 * r1 + 0.6 * g1 + 0.8 * b1
        img2 = -0.2 * r2 + 0.6 * g2 + 0.8 * b2
        # img = -0.0 * r + 0.4 * g + 0.6 * b
        cv2.imwrite(image_left_path, img1)
        cv2.imwrite(image_right_path, img2)
    return pinum

#获取自适应始波模板
def templatemade(srcpath,width,templatenum):
    imagefolderpath = srcpath
    onlyfiles = [f for f in listdir(imagefolderpath) if isfile(join(imagefolderpath, f))]
    onlyfiles.sort(key=lambda f: int(re.sub('\D', '', f)))
    # print (onlyfiles)
    imagepathlist = onlyfiles
    image0 = plt.imread(join(imagefolderpath, imagepathlist[templatenum]))
    img1 = image0.copy()
    #始波模板更新
    template1 = cv2.imread(r'C:\Users\Administrator\Desktop\pythonProject\image\template1.jpg', 0)
    h1, w1 = template1.shape[:2]
    #template1 = cv2.cvtColor(template1, cv2.COLOR_BGR2RGB)
    res1 = cv2.matchTemplate(img1, template1, cv2.TM_SQDIFF)
    min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
    top_left1 = min_loc1
    bottom_right1 = (top_left1[0] + w1, top_left1[1] + h1)
    # template3 = img1[top_left1[1]:bottom_right1[1],0:285]
    template3 = img1[top_left1[1]:bottom_right1[1], 0:143]
    cv2.imwrite(r'C:\Users\Administrator\Desktop\pythonProject\image\template3.jpg', template3)

    # img3 = img1[bottom_right1[1]:bottom_right1[1] + width, 0:285]
    img3 = img1[bottom_right1[1]:bottom_right1[1] + width, 0:143]
    cv2.imwrite(r'C:\Users\Administrator\Desktop\pythonProject\image\daichuli.jpg', img3)

    #底波模板更新
    img2 = plt.imread(r'C:\Users\Administrator\Desktop\pythonProject\image\daichuli.jpg')
    template2 = cv2.imread(r'C:\Users\Administrator\Desktop\pythonProject\image\template2.jpg', 0)
    h2, w2 = template2.shape[:2]
    res2 = cv2.matchTemplate(img3, template2, cv2.TM_SQDIFF)

    min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)
    top_left2 = min_loc2
    bottom_right2 = (top_left2[0] + w2, top_left2[1] + h2)
    # template4 = img2[top_left2[1]:bottom_right2[1], 0:285]
    template4 = img2[top_left2[1]:bottom_right2[1], 0:143]
    cv2.imwrite(r'C:\Users\Administrator\Desktop\pythonProject\image\template4.jpg', template4)

# 始波模板匹配筛除
def match_shibo(srcpath, dstpath, width):
    imagefolderpath = srcpath
    dstpath = dstpath
    onlyfiles = [f for f in listdir(imagefolderpath) if isfile(join(imagefolderpath, f))]
    onlyfiles.sort(key=lambda f: int(re.sub('\D', '', f)))
    # print (onlyfiles)
    imagepathlist = onlyfiles
    pinum = len(imagepathlist)
    for i in range(pinum):
        # print(i)
        image0 = plt.imread(join(imagefolderpath, imagepathlist[i]))
        dst_path = join(dstpath, imagepathlist[i])
        img2 = image0.copy()
        template = cv2.imread(r'C:\Users\Administrator\Desktop\pythonProject\image\template3.jpg', 0)
        # template = cv2.imread(r'C:\Users\Administrator\Desktop\pythonProject\image\template1.jpg', 0)
        h, w = template.shape[:2]
        res = cv2.matchTemplate(image0, template, cv2.TM_SQDIFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = min_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        # img3 = img2[bottom_right[1]:bottom_right[1] + width, 0:285]
        img3 = img2[bottom_right[1]:bottom_right[1] + width, 0:143]

        cv2.imwrite(dst_path, img3)

    return 0

#底波抓取+C扫描拼接
def divide_dibo(srcpath):
    imagefolderpath = srcpath
    # dibopath = dibopath
    onlyfiles = [f for f in listdir(imagefolderpath) if isfile(join(imagefolderpath, f))]
    onlyfiles.sort(key=lambda f: int(re.sub('\D', '', f)))
    # print (onlyfiles)
    imagepathlist = onlyfiles
    pinum = len(imagepathlist)
    dibo_list = []
    for i in range(pinum):
        #print(i)
        image0 = plt.imread(join(imagefolderpath, imagepathlist[i]))
        # dibo_path = join(dibopath, imagepathlist[i])
        img2 = image0.copy()
        template = cv2.imread(r'C:\Users\Administrator\Desktop\pythonProject\image\template4.jpg',0)
        # template = cv2.imread(r'C:\Users\Administrator\Desktop\pythonProject\image\template2.jpg', 0)
        h, w = template.shape[:2]
        res = cv2.matchTemplate(image0, template, cv2.TM_SQDIFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = min_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        # dibo = img2[top_left[1]:bottom_right[1],0:255]
        dibo = img2[top_left[1]:bottom_right[1], 0:143]
        dibo_list.append(dibo)
        #dibo = cv2.pyrDown(dibo)

        # cv2.imwrite(dibo_path, dibo)

    image1 = dibo_list[0]
    image1 = image1.astype(np.uint8)
    img0 = np.array(image1)
    [row, col] = img0.shape
    data_3d = np.zeros([pinum, col, 255], dtype='uint8')
    height = np.zeros((pinum, col), dtype='uint8')

    for m in range(pinum):
        image = dibo_list[m]
        image = 255 - image
        max = np.max(image)
        min = np.min(image)
        Omin, Omax = 0, 255
        a = float(Omax - Omin) / (max - min)
        b = Omin - a * min
        image = a * image + b
        image = image.astype(np.uint8)
        img = np.array(image)
        for a in range(col):
            sum = 0
            for b in range(row):
                if img[b][a] > 150:
                    sum = sum + img[b][a]
            h = int(round(sum / row))
            # if h > 25:
            # h = 25
            if h > 13:
                h = 13
            height[m][a] = h
            # data_3d[m, a, h] = 255
            for c in range(h):
                data_3d[m, a, c] = 255
    max = np.max(height)
    height = 255 / max * height
    height = height.astype(np.uint8)
    htest = height.swapaxes(0, 1)
    current_time = datetime.datetime.now()
    timename = str(current_time.strftime('%Y%m%d%H%M%S'))
    cv2.imwrite(join("C:/Users/Administrator/Desktop/pythonProject/result/"+'Y-'+timename + ".jpg"), htest)
    # cv_show('htest1', htest)

    htest = cv2.GaussianBlur(htest, (5, 5), sigmaX=1)
    # cv_show('htest2', htest)
    kernel = np.ones((7, 7), np.uint8)
    opening = cv2.morphologyEx(htest, cv2.MORPH_OPEN, kernel)
    # cv_show('htest3', opening)
    # cv2.imwrite("C:/Users/Administrator/Desktop/pythonProject/result/11111.jpg", opening)



    #淹膜操作
    htest2 = htest.copy()

    canny = cv2.Canny(opening, 0, 127)
    ret, thresh = cv2.threshold(canny, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    htest1 = cv2.cvtColor(htest, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(htest1, contours, -1, (0, 0, 255), 2)
    # cv_show('htest1', htest1)
    points = np.empty((0, 1, 2))
    [row, col] = htest.shape
    mask = np.zeros_like(htest)
    # cv_show('mask',mask)
    # for i in range(len(contours)):
    #     if 10 < cv2.contourArea(contours[i]):
    #         point = contours[i]
    #         x, y, w, h = cv2.boundingRect(point)
    #         # 左上角边界识别
    #         if y + h >= row-5 and x <= 5:
    #             points = np.vstack((points, point))
    #             [vx, vy, x1, y1] = cv2.fitLine(point, cv2.DIST_L2, 0, 0.01, 0.01)
    #
    #             lefty = int((-x1 * vy / vx) + y1)
    #             righty = int(((col - x1) * vy / vx) + y1)
    #             # htest2 = cv2.cvtColor(htest2, cv2.COLOR_GRAY2BGR)
    #             cv2.line(htest2, (col-1, righty), (0, lefty), (0, 0, 255), 2)
    #             # 淹膜制作
    #             triangle = np.array([(col - 1+10, righty), (0, lefty+10), (0, 0)])
    #             cv2.fillPoly(mask, [triangle], (255, 255, 255))
    #             print('111')
    #             # cv_show('mask', mask)
    #             # break
    #         #右上角边界识别
    #         if x + w >= col-5 and y <= 5:
    #             points = np.vstack((points, point))
    #             [vx, vy, x1, y1] = cv2.fitLine(point, cv2.DIST_L2, 0, 0.01, 0.01)
    #
    #             lefty = int((-x1 * vy / vx) + y1)
    #             righty = int(((col - x1) * vy / vx) + y1)
    #             cv2.line(htest, (col - 1, righty), (0, lefty), (0, 0, 255), 2)
    #             # 淹膜制作
    #             triangle1 = np.array([(col - 1, righty), (0, lefty), (col, 0)])
    #             cv2.fillPoly(mask, [triangle1], (255, 255, 255))
    #             print('222')
    #             # cv_show('mask2', mask)
    #         #左下角边界识别
    #         if x  == 0 and y == 0:
    #             points = np.vstack((points, point))
    #             [vx, vy, x1, y1] = cv2.fitLine(point, cv2.DIST_L2, 0, 0.01, 0.01)
    #             lefty = int((-x1 * vy / vx) + y1)
    #             righty = int(((col - x1) * vy / vx) + y1)
    #             cv2.line(htest, (col - 1, righty), (0, lefty), (0, 0, 255), 2)
    #             # 淹膜制作
    #             triangle2 = np.array([(col - 1, righty), (0, lefty), (0, row)])
    #             cv2.fillPoly(mask, [triangle2], (255, 255, 255))
    #             print('333')
    #         #右下角边界识别
    #         if y+h  == row and x+w == col:
    #             points = np.vstack((points, point))
    #             [vx, vy, x1, y1] = cv2.fitLine(point, cv2.DIST_L2, 0, 0.01, 0.01)
    #
    #             lefty = int((-x1 * vy / vx) + y1)
    #             righty = int(((col - x1) * vy / vx) + y1)
    #             cv2.line(htest, (col - 1, righty), (0, lefty), (0, 0, 255), 2)
    #             # 淹膜制作
    #             triangle3 = np.array([(col - 1, righty), (0, lefty), (col, row)])
    #             cv2.fillPoly(mask, [triangle3], (255, 255, 255))
    #             print('444')
    #             # cv_show('mask2', mask)

    # cv_show('mask2', mask)
    # cv_show('img', htest)
    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    dst = htest + mask
    # dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
    # cv_show('dst', dst)
    # dst = cv2.bitwise_and(img,mask)
    # cv_show('dst', dst)



    top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
    constant = cv2.copyMakeBorder(dst, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT, value=255)
    ret, thresh = cv2.threshold(constant, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    img3 = constant.copy()
    m = 0
    img4 = cv2.cvtColor(img3, cv2.COLOR_GRAY2BGR)
    for i in range(len(contours)):
        if 1< cv2.contourArea(contours[i]) < 100:
            x, y, w, h = cv2.boundingRect(contours[i])
            m = m + 1
            img3 = cv2.rectangle(img4, (x, y), (x + w, y + h), (0, 0, 255), 2)
    print(m)
    # cv_show('img', img3)
    # current_time = datetime.datetime.now()
    # timename = str(current_time.strftime('%Y%m%d%H%M%S'))
    # cv2.imwrite(join(timename + ".jpg"), htest)
    if m>=1:
        cv2.imwrite(join("C:/Users/Administrator/Desktop/pythonProject/result/"+'D-'+timename + ".jpg"), img3)
        file = join("C:/Users/Administrator/Desktop/pythonProject/result/"+'D-'+timename + ".jpg")
    else:
        cv2.imwrite(join("C:/Users/Administrator/Desktop/pythonProject/result/" + timename  + ".jpg"), img3)
        file = join("C:/Users/Administrator/Desktop/pythonProject/result/" + timename  + ".jpg")
    return m , file

def pinjie(return1,return2):
    num1,file_path1 = return1
    num1 = int(num1)
    img1 = cv2.imread(join(file_path1))
    num2, file_path2 = return2
    num2 = int(num2)
    img2 = cv2.imread(join(file_path2))
    img3 = np.vstack((img1,img2))
    # cv_show('img3',img3)
    current_time = datetime.datetime.now()
    timename = str(current_time.strftime('%Y%m%d%H%M%S'))
    if num1+num2>=1:
        cv2.imwrite(join("C:/Users/Administrator/Desktop/pythonProject/result/" + 'P-D-' + timename + ".jpg"), img3)
        file = join("C:/Users/Administrator/Desktop/pythonProject/result/" + 'P-D-' + timename + ".jpg")
    else:
        cv2.imwrite(join("C:/Users/Administrator/Desktop/pythonProject/result/" +'P-'+ timename  + ".jpg"), img3)
        file = join("C:/Users/Administrator/Desktop/pythonProject/result/" + 'P-'+timename  + ".jpg")

    return num1,num2,file







def main(i):
    clear_folder(r'C:\Users\Administrator\Desktop\pythonProject\image\YCLleft')
    clear_folder(r'C:\Users\Administrator\Desktop\pythonProject\image\YCLright')
    clear_folder(r'C:\Users\Administrator\Desktop\pythonProject\image\fullleft')
    clear_folder(r'C:\Users\Administrator\Desktop\pythonProject\image\fullright')
    pinum = pre(r'C:\Users\Administrator\Desktop\pic',
        r'C:\Users\Administrator\Desktop\pythonProject\image\YCLleft',
        r'C:\Users\Administrator\Desktop\pythonProject\image\YCLright')
    pinum = int(pinum/2)
    templatemade(r'C:\Users\Administrator\Desktop\pythonProject\image\YCLleft', i, pinum)
    match_shibo(r'C:\Users\Administrator\Desktop\pythonProject\image\YCLleft',
                r'C:\Users\Administrator\Desktop\pythonProject\image\fullleft', i)
    D_l = divide_dibo(r'C:\Users\Administrator\Desktop\pythonProject\image\fullleft')
    # print(D_l)

    templatemade(r'C:\Users\Administrator\Desktop\pythonProject\image\YCLright', i, pinum)
    match_shibo(r'C:\Users\Administrator\Desktop\pythonProject\image\YCLright',
                r'C:\Users\Administrator\Desktop\pythonProject\image\fullright', i)
    # divide_dibo(r'C:\Users\Administrator\Desktop\pythonProject\image\fullright')
    D_r = divide_dibo(r'C:\Users\Administrator\Desktop\pythonProject\image\fullright')
    m = pinjie(D_l, D_r)
    print(m)
    # print(D_r)
    return m

def test(x):
    time.sleep(10)
    return x+1,x+2

# print(test(1))

def main1(i):
    try:
        clear_folder(r'C:\Users\Administrator\Desktop\pythonProject\image\YCLleft')
        clear_folder(r'C:\Users\Administrator\Desktop\pythonProject\image\YCLright')
        clear_folder(r'C:\Users\Administrator\Desktop\pythonProject\image\fullleft')
        clear_folder(r'C:\Users\Administrator\Desktop\pythonProject\image\fullright')
        pinum = pre(r'C:\Users\Administrator\Desktop\pic',
                    r'C:\Users\Administrator\Desktop\pythonProject\image\YCLleft',
                    r'C:\Users\Administrator\Desktop\pythonProject\image\YCLright')
        pinum = int(pinum / 2)
        templatemade(r'C:\Users\Administrator\Desktop\pythonProject\image\YCLleft', i, pinum)
        match_shibo(r'C:\Users\Administrator\Desktop\pythonProject\image\YCLleft',
                    r'C:\Users\Administrator\Desktop\pythonProject\image\fullleft', i)
        D_l = divide_dibo(r'C:\Users\Administrator\Desktop\pythonProject\image\fullleft')
        # print(D_l)

        templatemade(r'C:\Users\Administrator\Desktop\pythonProject\image\YCLright', i, pinum)
        match_shibo(r'C:\Users\Administrator\Desktop\pythonProject\image\YCLright',
                    r'C:\Users\Administrator\Desktop\pythonProject\image\fullright', i)
        # divide_dibo(r'C:\Users\Administrator\Desktop\pythonProject\image\fullright')
        D_r = divide_dibo(r'C:\Users\Administrator\Desktop\pythonProject\image\fullright')
        m = pinjie(D_l, D_r)
        print(m)
        # print(D_r)
        return m
    except:
        return -1,-1,'-1'

if __name__ == '__main__':
    x= main1(150)
    print(x)
# main(150)