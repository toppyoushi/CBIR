import Pretreatment as pt
import Color as clr
import Texture as txtr
import Shape as shp
import numpy as np
import cv2 as cv
import os

fpath = os.path.join(os.getcwd(),'FeatureSet/path.txt')
colorjuPath=os.path.join(os.getcwd(),'FeatureSet/colorjuData.txt')
greymatrixPath=os.path.join(os.getcwd(),'FeatureSet/greymatrixData.txt')
shapeHistogramPath=os.path.join(os.getcwd(),'FeatureSet/shapeHistogramData.txt')
shapeNchangePath=os.path.join(os.getcwd(),'FeatureSet/shapeNchangeData.txt')
# f = open(fpath,'w')
# f1 = open(colorjuPath,'w')
f2 = open(greymatrixPath,'w')
# f3 = open(shapeHistogramPath,'w')
# f4 = open(shapeNchangePath,'w')
filepath = os.path.join(os.getcwd(),'ImageSet')
class_set=os.listdir(filepath)
for i in class_set:
    img_set=os.listdir(os.path.join(filepath,i))
    for j in img_set:
        img_path = os.path.join(os.path.join(filepath,i),j)
        # f.write(img_path+'\n')
        # color_feature = clr.calColorFeature(img_path)
        grey_matrix_feature = txtr.calTextureFeature(img_path)
        # shapeHistogram_feature = shp.calShapeHistogram(img_path)
        # shapeNchage_feature = shp.calHuMoment(img_path)
        # print(j)
        # f1.write(str(color_feature)+'\n')
        f2.write(str(grey_matrix_feature)+'\n')
        # f3.write(str(shapeHistogram_feature)+'\n')
        # f4.write(str(shapeNchage_feature)+'\n')
# f.close()
# f1.close()
f2.close()
# f3.close()
# f4.close()
        
