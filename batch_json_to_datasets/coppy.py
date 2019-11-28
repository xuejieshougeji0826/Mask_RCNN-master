#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
# author:"Zhang Shuyu"
"""使用skimage模块读取图片，不改变图片数据类型uint16，保存为uint8类型"""
import os,shutil
import cv2
import natsort
import numpy as np
from skimage import io
from matplotlib import pylab as plt
print("???")
input_file = r"C:\Users\MSI\Desktop\mask_cnn_test-pics\train_data\label_json\\"  #文件路径
img_type = ".png"
output_file = r"C:\Users\MSI\Desktop\mask_cnn_test-pics\train_data\cv2_mask"
for root, dirs, files in os.walk(input_file,topdown=True):
    for name in natsort.natsorted(dirs):  #natsort,自然排序
        print(name)
        file_name = os.path.join(input_file + name,"label" + img_type)
        img = io.imread(file_name)
        print(file_name)
        midname = name[:name.rindex("_")]
        # print()
        cv2.imwrite(os.path.join(output_file, midname + img_type),img) # 路径必须是英文才能正常保存
