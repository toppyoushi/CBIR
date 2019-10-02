import cv2 as cv
import numpy as np
import Pretreatment as pt
import sklearn.preprocessing as prp
import math
gray_level = 8
gray_scale = [0, 32, 64, 96, 128, 160, 192, 224, 256]


def feature_computer(p):
    Con = 0.0  # 纹理对比度
    Ent = 0.0  # 纹理熵
    Asm = 0.0  # 能量
    Idm = 0.0  # 纹理相关性
    for i in range(gray_level):
        for j in range(gray_level):
            Con += (i-j)*(i-j)*p[i][j]
            Asm += p[i][j]*p[i][j]
            Idm += p[i][j]/(1+(i-j)*(i-j))
            if p[i][j] > 0.0:
                Ent += p[i][j]*math.log(p[i][j])
    return Asm, Con, -Ent, Idm


def calGLCM(matrix):
    '''
    计算0度、45度、90度、135度的灰度共生矩阵
    '''
    grey_matrix_0 = np.zeros((len(gray_scale)-1, len(gray_scale)-1), dtype=int)
    grey_matrix_45 = np.zeros(
        (len(gray_scale)-1, len(gray_scale)-1), dtype=int)
    grey_matrix_90 = np.zeros(
        (len(gray_scale)-1, len(gray_scale)-1), dtype=int)
    grey_matrix_135 = np.zeros(
        (len(gray_scale)-1, len(gray_scale)-1), dtype=int)
    for indexi, i in enumerate(matrix):
        for indexj, j in enumerate(i[:-1]):
            neigh_0 = (int)(matrix[indexi, indexj+1])
            grey_matrix_0[(int)(j), neigh_0] += 1
    for indexi, i in enumerate(matrix[1:]):
        for indexj, j in enumerate(i[:-1]):
            neigh_45 = (int)(matrix[indexi-1, indexj+1])
            grey_matrix_45[(int)(j), neigh_45] += 1
    for indexi, i in enumerate(matrix[1:]):
        for indexj, j in enumerate(i):
            neigh_90 = (int)(matrix[indexi-1, indexj])
            grey_matrix_90[(int)(j), neigh_90] += 1
    for indexi, i in enumerate(matrix[1:]):
        for indexj, j in enumerate(i[1:]):
            neigh_135 = (int)(matrix[indexi-1, indexj-1])
            grey_matrix_135[(int)(j), neigh_135] += 1
    grey_matrix_0=prp.scale(grey_matrix_0)
    grey_matrix_45=prp.scale(grey_matrix_45)
    grey_matrix_90=prp.scale(grey_matrix_90)
    grey_matrix_135=prp.scale(grey_matrix_135)
    print(grey_matrix_0, grey_matrix_45, grey_matrix_90, grey_matrix_135)
    return grey_matrix_0, grey_matrix_45, grey_matrix_90, grey_matrix_135


def calTextureFeature(filename):
    img = cv.imread(filename, cv.IMREAD_COLOR)
    if img is None:
        return
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (400, 300))  # 统一图像大小
    reduced_img = np.zeros((img.shape[0], img.shape[1]))
    for indexi, i in enumerate(img):
        for indexj, j in enumerate(i):
            for indexk, k in enumerate(gray_scale[:-1]):
                if j >= k and j < gray_scale[indexk+1]:
                    reduced_img[indexi, indexj] = indexk
    grey_matrix_0, grey_matrix_45, grey_matrix_90, grey_matrix_135 = calGLCM(
        reduced_img)
    asm_0, con_0, eng_0, idm_0 = feature_computer(grey_matrix_0)
    asm_45, con_45, eng_45, idm_45 = feature_computer(grey_matrix_45)
    asm_90, con_90, eng_90, idm_90 = feature_computer(grey_matrix_90)
    asm_135, con_135, eng_135, idm_135 = feature_computer(grey_matrix_135)
    asm_mean = np.mean([asm_0, asm_45, asm_90, asm_135])
    asm_var = np.var([asm_0, asm_45, asm_90, asm_135])
    con_mean = np.mean([con_0, con_45, con_90, con_135])
    con_var = np.var([con_0, con_45, con_90, con_135])
    eng_mean = np.mean([eng_0, eng_45, eng_90, eng_135])
    eng_var = np.var([eng_0, eng_45, eng_90, eng_135])
    idm_mean = np.mean([idm_0, idm_45, idm_90, idm_135])
    idm_var = np.var([idm_0, idm_45, idm_90, idm_135])
    print(asm_mean, asm_var, con_mean, con_var,
          eng_mean, eng_var, idm_mean, idm_var)
    return [asm_mean, asm_var, con_mean, con_var, eng_mean, eng_var, idm_mean, idm_var]
