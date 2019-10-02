import numpy as np
import cv2 as cv
import Pretreatment as pt
import sklearn.preprocessing as prp
import math


def calHuMoment(filename):
    '''计算图像Hu矩'''
    img = cv.imread(filename, cv.IMREAD_COLOR)
    if img is None:
        return
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (400, 300))
    img = cv.medianBlur(img, 3)
    moments = cv.moments(img)
    humoments = cv.HuMoments(moments)
    humoments = humoments.T
    humoments_list = []
    for i in range(len(humoments)):
        humoments_list.extend(humoments[i])
    return humoments_list


def calShapeHistogram(filename):
    '''计算图像形状边缘直方图特征'''
    img = cv.imread(filename, cv.IMREAD_COLOR)
    if img is None:
        return
    img_gaussian = cv.GaussianBlur(img, (5, 5), 1)  # 第一步高斯滤波处理
    img_gaussian = cv.cvtColor(img_gaussian, cv.COLOR_BGR2GRAY)  # RGB转灰度图
    img_gaussian = cv.resize(img_gaussian, (400, 300))
    w, h = img_gaussian.shape
    dx = np.zeros((w-1, h-1))
    dy = np.zeros((w-1, h-1))
    d = np.zeros((w-1, h-1))
    for i in range(w-1):  # 第二步计算梯度幅值
        for j in range(h-1):
            dx[i, j] = int(img_gaussian[i, j+1])-int(img_gaussian[i, j])
            dy[i, j] = int(img_gaussian[i+1, j])-int(img_gaussian[i, j])
            d[i, j] = math.sqrt(dx[i, j]**2+dy[i, j]**2)
    w1, h1 = d.shape
    nms = np.zeros((w1-1, h1-1))
    for i in range(1, w1-1):  # 第三步非极大值抑制
        for j in range(1, h1-1):
            if d[i, j] == 0:
                nms[i, j] = 0
            else:
                gradX = dx[i, j]
                gradY = dy[i, j]
                grad = d[i, j]
                if np.abs(gradY) > np.abs(gradX):
                    weight = np.abs(gradX)/np.abs(gradY)
                    grad2 = d[i-1, j]
                    grad4 = d[i+1, j]
                    if gradX*gradY > 0:
                        grad1 = d[i-1, j-1]
                        grad3 = d[i+1, j+1]
                    else:
                        grad1 = d[i-1, j+1]
                        grad3 = d[i+1, j-1]
                else:
                    weight = np.abs(gradY)/np.abs(gradX)
                    grad2 = d[i, j-1]
                    grad4 = d[i, j+1]
                    if gradX*gradY > 0:
                        grad1 = d[i+1, j-1]
                        grad3 = d[i-1, j+1]
                    else:
                        grad1 = d[i-1, j-1]
                        grad3 = d[i+1, j+1]
                dTemp1 = weight*grad1 + (1-weight)*grad2
                dTemp2 = weight*grad3 + (1-weight)*grad4
                if grad > dTemp1 and grad > dTemp2:
                    nms[i, j] = grad
                else:
                    nms[i, j] = 0
    w2, h2 = nms.shape
    new_nms = np.zeros((w2, h2))
    tl = 0.2*np.max(nms)
    th = 0.3*np.max(nms)
    for i in range(1, w2-1):  # 第四步双阈值检测、边缘连接
        for j in range(1, h2-1):
            if nms[i, j] < tl:  # 应当被抑制
                new_nms[i, j] = 0
            elif nms[i, j] > th:  # 强边缘点
                new_nms[i, j] = 1
            elif (nms[i-1, j-1:j+1] > th).any or (nms[i+1, j-1:j+1] > th).any or (nms[i, j-1] > th) or (nms[i, j+1] > th):
                new_nms[i, j] = 1   # 孤立弱边缘点
    new_nms_w = new_nms.shape[0]
    new_nms_w_d = new_nms_w/4
    new_nms_h = new_nms.shape[1]
    new_nms_h_d = new_nms_h/4
    histogram = [0.0]*16
    histogram_sum = 0
    for i in range(new_nms_w):  # 统计边缘直方图的特征
        for j in range(new_nms_h):
            if new_nms[i, j] == 1:
                if i >= 0 and i < new_nms_w_d:
                    if j >= 0 and j < new_nms_h_d:
                        histogram[0] += 1
                    if j >= new_nms_h_d and j < 2*new_nms_h_d:
                        histogram[1] += 1
                    if j >= 2*new_nms_h_d and j < 3*new_nms_h_d:
                        histogram[2] += 1
                    if j >= 3*new_nms_h_d and j < 4*new_nms_h_d:
                        histogram[3] += 1
                if i >= new_nms_w_d and i < 2*new_nms_w_d:
                    if j >= 0 and j < new_nms_h_d:
                        histogram[4] += 1
                    if j >= new_nms_h_d and j < 2*new_nms_h_d:
                        histogram[5] += 1
                    if j >= 2*new_nms_h_d and j < 3*new_nms_h_d:
                        histogram[6] += 1
                    if j >= 3*new_nms_h_d and j < 4*new_nms_h_d:
                        histogram[7] += 1
                if i >= 2*new_nms_w_d and i < 3*new_nms_w_d:
                    if j >= 0 and j < new_nms_h_d:
                        histogram[8] += 1
                    if j >= new_nms_h_d and j < 2*new_nms_h_d:
                        histogram[9] += 1
                    if j >= 2*new_nms_h_d and j < 3*new_nms_h_d:
                        histogram[10] += 1
                    if j >= 3*new_nms_h_d and j < 4*new_nms_h_d:
                        histogram[11] += 1
                if i >= 3*new_nms_w_d and i < 4*new_nms_w_d:
                    if j >= 0 and j < new_nms_h_d:
                        histogram[12] += 1
                    if j >= new_nms_h_d and j < 2*new_nms_h_d:
                        histogram[13] += 1
                    if j >= 2*new_nms_h_d and j < 3*new_nms_h_d:
                        histogram[14] += 1
                    if j >= 3*new_nms_h_d and j < 4*new_nms_h_d:
                        histogram[15] += 1
                histogram_sum += 1
    for i in range(len(histogram)):
        histogram[i] = histogram[i]/histogram_sum
    return histogram

    # th = np.amax(img)
    # tl = np.amin(img)
    # img = cv.Canny(img,th,tl)
    # cv.imshow('g_blur',img)
    # cv.waitKey(0)

    # dx = cv.Sobel(img, cv.CV_16S, 1, 0)                                  #第二步计算梯度
    # dy = cv.Sobel(img, cv.CV_16S, 0, 1)
    # absX = cv.convertScaleAbs(dx)    # 转回uint8
    # absY = cv.convertScaleAbs(dy)
    # w,h = img.shape
    # for i in range(1,w-1):
    #     for j in range(1,h-1):
    #         grad=dy[i,j]/dx[i,j]

    # dst = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

    # d_angle = np.zeros(absX.shape)
    # for indexi, i in enumerate(absX):
    #     for indexj, j in enumerate(i):
    #         d_angle[indexi, indexj] = (
    #            180*math.atan2(j, absY[indexi, indexj])/math.pi) % 360
    # cv.imshow('d_angle',d_angle)
    # cv.waitKey(0)
