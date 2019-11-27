#! /usr/bin/env python
# coding=utf-8
import os
import shutil
import time
import sys
import importlib
importlib.reload(sys)

def copy_and_rename(fpath_input, fpath_output):
    for file in os.listdir(fpath_input):
        for inner in os.listdir(fpath_input+file+'/'):
            print(inner)
            if os.path.splitext(inner)[0] == "label":
                former = os.path.join(fpath_input, file)
                oldname = os.path.join(former, inner)
                print(oldname)
                newname_1 = os.path.join(fpath_output,
                                         file.split('_')[0] + ".png")
                # os.rename(oldname, newname_1)
                print(newname_1)
                shutil.copyfile(oldname, newname_1)
                os.rename("./train_data/cv2_mask\pic.png","./train_data/cv2_mask/"+oldname[26:34]+".png")


if __name__ == '__main__':
    print('start ...')
    t1 = time.time() * 1000
    #time.sleep(1) #1s
    fpath_input = "./train_data/labelme_json/" #...为train_data文件夹地址，按自己的地址修改
    fpath_output = "./train_data/cv2_mask/"
    copy_and_rename(fpath_input, fpath_output)
    t2 = time.time() * 1000
    print('take time:' + str(t2 - t1) + 'ms')
    print('end.')

# os.rename("./train_data/cv2_mask\pic.png","./train_data/cv2_mask\pic2.png")