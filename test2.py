import cv2 
from cv2 import cv2 # if you use vscode it will cancel the error in cv2


img=cv2.imread(r"C:\Users\MSI\Desktop\Mask_RCNN-master\train_data\labelme_json\pic_0455_json/img.png")
print(img)
cv2.imshow("1",img)
cv2.waitKey(0)