import numpy as np
import cv2 as cv
def prtrtmnt(img):
    unify_img=cv.resize(img,(400,300))                                      #将图像统一为250x250大小
    # cv.imshow('unify_img',unify_img)
    # cv.waitKey(0)
    gauss_img=cv.GaussianBlur(unify_img,(7,7),0,borderType=cv.BORDER_WRAP)    #对图像进行高斯滤波处理消除噪声
    # cv.imshow('gauss_img',gauss_img)
    # cv.waitKey(0)
    equalized_img=cv.equalizeHist(gauss_img)                                #对图像均衡化处理，增强对比图
    # cv.imshow('equalized_img',equalized_img)
    # cv.waitKey(0)
    return equalized_img
def normalize(data):        #矩阵归一化
    m = np.mean(data)
    mx = max(data)
    mn = min(data)
    return [(float(i) - m) / (mx - mn) for i in data]
    