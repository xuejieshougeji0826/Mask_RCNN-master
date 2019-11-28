#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
# author:"Zhang Shuyu"
"""使用skimage模块读取图片，不改变图片数据类型uint16，保存为uint8类型"""
import os
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
        midname = name[:name.rindex("_")]
        img = io.imread(file_name)  #Todo:使用skimage模块读取图片不会改变图片的数据格式
        img = np.array(plt.imread(file_name)) #TODO:不要使用plt.imread读取图片，因为会改变图片的数据格式，uint16读入后会变成float32
        # print(img.shape)
        # io.imshow(img)
        # io.show()
        # plt.imshow(img * 10000) #Todo：img乘以10000是因为uint16位是 0--65531，在MATLAB直接显示时为黑色；标签是1-6
        # plt.axis('off')
        # plt.show()
        print(img.dtype)
        img = img.astype(np.uint8)
        print(img.dtype)
        cv2.imwrite(os.path.join(output_file, midname + img_type),img) # 路径必须是英文才能正常保存
        # plt.imshow(img * 40) #Todo：img乘以40是因为uint16位是 0--255，在MATLAB直接显示时为黑色 ；标签是1-6
        # plt.show()