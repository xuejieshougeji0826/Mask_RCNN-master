import cv2


img=cv2.imread("../train_data/labelme_json/pic_0455_json/img.png")
print(img)
cv2.imshow("1",img)
cv2.waitKey(0)