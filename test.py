import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
# Root directory of the project
ROOT_DIR = os.path.abspath("../")
from samples.coco import coco
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "coco/"))  # To find local version
# import coco



# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR)

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MODEL_DIR ,"mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    print("cuiwei***********************")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
# Load a random image from the images folder
#file_names = next(os.walk(IMAGE_DIR))[2]
#image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
cap = cv2.VideoCapture(0)

# while(1):
#     # get a frame
#     ret, frame = cap.read()
#     # show a frame
#     image = cv2.imread(r"C:\\Users\\MSI\\Desktop\\毕业代码\\mask_R_CNN\\Mask_RCNN\\images\\9247489789_132c0d534a_z.jpg")
#     print("frame:",frame)
#     # print(frame.shape)
#     print("frame_type_",type(frame))
#     print("frame:",image)
#     # print(image.shape)
#     print("image_type_",type(image))
#     start =time.clock()
#     results = model.detect([frame], verbose=1)
#     r = results[0]
#     cv2.imshow("capture", frame)
#     visualize.display_instances(frame, r['rois'], r['masks'], r['class_ids'],
#                             class_names, r['scores'])
#     end = time.clock()
#     # cv2.waitKey(0)
#     print(end-start)
#
#     key = cv2.waitKey(0) & 0xFF
#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
start =time.clock()
image= cv2.imread(r"C:\Users\MSI\Desktop\mask_cnn_test-pics\test\2.jpg")
# Run detection
b,g,r=cv2.split(image)
image2 = cv2.merge([r,g,b])
results = model.detect([image], verbose=1)
print(results)
end = time.clock()
print(end-start)
# Visualize results
r = results[0]
visualize.display_instances(image2, r['rois'], r['masks'], r['class_ids'],
                           class_names, r['scores'])
