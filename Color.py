import math
import numpy as np
import cv2 as cv


def RGB2HSI(img):
    '''将图像从RGB通道转化为HSI通道'''
    row = img.shape[0]
    col = img.shape[1]
    img_hsi = np.zeros((row, col, 3))
    img_b = img[:, :, 0]/255.0
    img_g = img[:, :, 1]/255.0
    img_r = img[:, :, 2]/255.0
    for indexi, i in enumerate(img_b):
        for indexj, j in enumerate(i):
            b = img_b[indexi, indexj]
            g = img_g[indexi, indexj]
            r = img_r[indexi, indexj]
            numerator = (r-g)+(r-b)
            n = (r-g)**2+(r-b)*(g-b)
            denominator = 2*math.sqrt(n)
            if denominator == 0.0:
                hue = 0.0
                saturation = 0.0
            else:
                theta = math.acos(numerator/denominator)
                if g < b:
                    hue = 2*math.pi-theta
                else:
                    hue = theta
                saturation = 1.0-(3.0*min((r, g, b))/(r+g+b))
            intensity = (r+g+b)/3.0
            img_hsi[indexi, indexj] = [hue, saturation, intensity]
    return img_hsi
    # return cv.cvtColor(img,cv.COLOR_BGR2HSV)


def calColorFeature(filename):
    '''计算图像的HSI特征值'''
    img = cv.imread(filename, cv.IMREAD_COLOR)
    if img is None:
        return
    color_feature = []
    unify_img = cv.resize(img, (400, 300))  # 将图像统一为400x300大小
    gauss_img = cv.GaussianBlur(
        unify_img, (5, 5), 0, borderType=cv.BORDER_WRAP)  # 对图像进行高斯滤波处理消除噪声
    img_hsi = RGB2HSI(img)                  # 将图像从RGB通道转换为HSI通道
    h, s, i = cv.split(img_hsi)
    #一阶中心距
    h_mean = np.mean(h)
    s_mean = np.mean(s)
    i_mean = np.mean(i)
    color_feature.extend([h_mean, s_mean, i_mean])
    #二阶中心矩
    h_std = np.std(h)
    s_std = np.std(s)
    i_std = np.std(i)
    color_feature.extend([h_std, s_std, i_std])
    #三阶中心矩
    h_skewness = np.mean(abs(h - h.mean())**3)
    s_skewness = np.mean(abs(s - s.mean())**3)
    i_skewness = np.mean(abs(i - i.mean())**3)
    h_thirdMoment = h_skewness**(1./3)
    s_thirdMoment = s_skewness**(1./3)
    i_thirdMoment = i_skewness**(1./3)
    color_feature.extend([h_thirdMoment, s_thirdMoment, i_thirdMoment])
    return color_feature
