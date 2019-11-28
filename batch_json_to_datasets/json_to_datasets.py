
import os
path = r'C:\Users\MSI\Desktop\mask_cnn_test-pics\train_data\json'  # path为json文件存放的路径
json_file = os.listdir(path)
for file in json_file:
    os.system("python E:\python\Scripts/labelme_json_to_dataset.exe %s"%(path + '/' + file))
