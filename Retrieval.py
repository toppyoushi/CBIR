import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2 as cv
from PIL import Image, ImageTk
import math
import Color as clr
import Texture as txtr
import Shape as shp

file = open('./FeatureSet/path.txt', 'r')
color_moment = open('./FeatureSet/colorjuData.txt', 'r')
grey_matrix = open('./FeatureSet/greymatrixData.txt', 'r')
shape_histogram = open('./FeatureSet/shapeHistogramData.txt', 'r')
hu_moment = open('./FeatureSet/shapeNchangeData.txt', 'r')
path_file = file.readlines()
color_list = color_moment.readlines()
texture_list = grey_matrix.readlines()
hu_moment_list = hu_moment.readlines()
shape_histogram_list = shape_histogram.readlines()
file.close()
color_moment.close()
grey_matrix.close()
shape_histogram.close()
hu_moment.close()


def selectImage():
    filepath = filedialog.askopenfilename(title='选择待检索图片')
    return filepath


def calEuclidDistance(v1, v2):
    '''
    计算欧式距离
    '''
    d = 0.0
    for i in v1-v2:
        d += i**2
    d = math.sqrt(d)
    return d


def calManhattanDistance(v1, v2):
    '''
    计算曼哈顿距离
    '''
    d = 0.0
    for i in range(len(v1)):
        d += np.abs(v1[i]-v2[i])
    return d


def crossMeasureFunc(v1, v2):
    '''
    相交法度量函数
    '''
    a = 0.0
    b = 0.0
    for i in range(len(v1)):
        a += min((v1[i], v2[i]))
        b += v1[i]
    d = a/b
    return d


def searchByColor():
    '''
    中心矩法，相似度为计算欧式距离
    '''
    filepath = selectImage()
    if not filepath:
        return
    new_color_moment = np.array(clr.calColorFeature(filepath))
    distance = []
    for i in range(len(color_list)):
        line = color_list[i][1:-2].split(',')
        for j in range(len(line)):
            line[j] = float(line[j])
        line = np.array(line)
        d = calEuclidDistance(new_color_moment, line)
        distance.append((i, d))
    distance = sorted(distance, key=lambda x: x[1])
    distance = distance[0:16]
    neigh = []
    for i in distance:
        neigh.append(path_file[i[0]][:-1])
    new_img = plt.imread(filepath)
    new_img = cv.resize(new_img, (500, 300))
    f, ax = plt.subplots(4, 5, figsize=(10, 6))
    ax[0, 0].set_title('input img')
    ax[0, 0].imshow(new_img)
    ax[0, 0].axis('off')
    ax[1, 0].axis('off')
    ax[2, 0].axis('off')
    ax[3, 0].axis('off')
    for i in range(len(neigh)):
        axisX = int(i/4)
        axisY = int(i % 4)+1
        # print(axisX,axisY)
        temp_img = cv.resize(plt.imread(neigh[i]), (500, 300))
        ax[axisX, axisY].set_title('distance:'+str(round(distance[i][1], 2)))
        ax[axisX, axisY].imshow(temp_img)
        ax[axisX, axisY].axis('off')
    plt.show()


def searchByTexture():
    '''
    灰度共生矩阵，相似度为计算欧式距离
    '''
    filepath = selectImage()
    if not filepath:
        return
    new_grey_matrix = np.array(txtr.calTextureFeature(filepath))
    distance = []
    for i in range(len(texture_list)):
        line = texture_list[i][1:-2].split(',')
        for j in range(len(line)):
            line[j] = float(line[j])
        line = np.array(line)
        d = calEuclidDistance(new_grey_matrix, line)
        distance.append((i, d))
    distance = sorted(distance, key=lambda x: x[1])
    distance = distance[0:16]
    neigh = []
    for i in distance:
        neigh.append(path_file[i[0]][:-1])
    new_img = plt.imread(filepath)
    new_img = cv.resize(new_img, (500, 300))
    f, ax = plt.subplots(4, 5, figsize=(10, 6))
    ax[0, 0].set_title('input img')
    ax[0, 0].imshow(new_img)
    ax[0, 0].axis('off')
    ax[1, 0].axis('off')
    ax[2, 0].axis('off')
    ax[3, 0].axis('off')
    for i in range(len(neigh)):
        axisX = int(i/4)
        axisY = int(i % 4)+1
        temp_img = cv.resize(plt.imread(neigh[i]), (500, 300))
        ax[axisX, axisY].set_title('distance:'+str(round(distance[i][1], 2)))
        ax[axisX, axisY].imshow(temp_img)
        ax[axisX, axisY].axis('off')
    plt.show()


def searchByShapeNChange():
    '''
    旋转不变矩法
    '''
    filepath = selectImage()
    if not filepath:
        return
    new_hu_moment = np.array(shp.calHuMoment(filepath))
    distance = []
    for i in range(len(hu_moment_list)):
        line = hu_moment_list[i][1:-2].split(',')
        for j in range(len(line)):
            line[j] = float(line[j])
        line = np.array(line)
        d = calManhattanDistance(new_hu_moment, line)
        distance.append((i, d))
    distance = sorted(distance, key=lambda x: x[1])
    distance = distance[0:16]
    print(distance)
    neigh = []
    for i in distance:
        neigh.append(path_file[i[0]][:-1])
    new_img = plt.imread(filepath)
    new_img = cv.resize(new_img, (500, 300))
    f, ax = plt.subplots(4, 5, figsize=(10, 6))
    ax[0, 0].set_title('input img')
    ax[0, 0].imshow(new_img)
    ax[0, 0].axis('off')
    ax[1, 0].axis('off')
    ax[2, 0].axis('off')
    ax[3, 0].axis('off')
    for i in range(len(neigh)):
        axisX = int(i/4)
        axisY = int(i % 4)+1
        temp_img = cv.resize(plt.imread(neigh[i]), (500, 300))
        ax[axisX, axisY].set_title('distance:'+str(round(distance[i][1], 2)))
        ax[axisX, axisY].imshow(temp_img)
        ax[axisX, axisY].axis('off')
    plt.show()


def searchByShapeHistogram():
    '''
    直方图相交法，相似度为计算直方图度量函数
    '''
    filepath = selectImage()
    if not filepath:
        return
    new_shape_histogram = np.array(shp.calShapeHistogram(filepath))
    likelihood = []
    for i in range(len(shape_histogram_list)):
        line = shape_histogram_list[i][1:-2].split(',')
        for j in range(len(line)):
            line[j] = float(line[j])
        line = np.array(line)
        d = crossMeasureFunc(new_shape_histogram, line)
        likelihood.append((i, d))
    likelihood = sorted(likelihood, key=lambda x: x[1],reverse=True)
    likelihood = likelihood[0:16]
    neigh = []
    for i in likelihood:
        neigh.append(path_file[i[0]][:-1])
    new_img = plt.imread(filepath)
    new_img = cv.resize(new_img, (500, 300))
    f, ax = plt.subplots(4, 5, figsize=(10, 6))
    ax[0, 0].set_title('input img')
    ax[0, 0].imshow(new_img)
    ax[0, 0].axis('off')
    ax[1, 0].axis('off')
    ax[2, 0].axis('off')
    ax[3, 0].axis('off')
    for i in range(len(neigh)):
        axisX = int(i/4)
        axisY = int(i % 4)+1
        temp_img = cv.resize(plt.imread(neigh[i]), (500, 300))
        ax[axisX, axisY].set_title('likelihood:'+str(round(likelihood[i][1], 2)))
        ax[axisX, axisY].imshow(temp_img)
        ax[axisX, axisY].axis('off')
    plt.show()



win = tk.Tk()
win.title('基于内容的图像检索系统')
win.geometry('400x300+400+200')

b1 = tk.Button(win, text='HSI中心矩法', command=searchByColor)
b2 = tk.Button(win, text='灰度共生矩阵法', command=searchByTexture)
b3 = tk.Button(win, text='形状不变矩法', command=searchByShapeNChange)
b4 = tk.Button(win, text='直方图相交法', command=searchByShapeHistogram)


b1.pack(fill=tk.X, anchor=tk.CENTER, pady=10)
b2.pack(fill=tk.X, anchor=tk.CENTER, pady=10)
b3.pack(fill=tk.X, anchor=tk.CENTER, pady=10)
b4.pack(fill=tk.X, anchor=tk.CENTER, pady=10)
win.mainloop()
