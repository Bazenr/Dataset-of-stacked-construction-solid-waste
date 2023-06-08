#%%

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)

# Import Mask RCNN

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from samples.cdw import cdw


# To find local version of the library

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
os.environ['CUDA_VISIBLE_DEVICES']='1'
# Local path to trained weights file
# CDW_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_cdw_0020.h5")
CDW_MODEL_PATH = 'E:/20201203/model/rgbh/logs/cdw20210125T1634/mask_rcnn_cdw_0120.h5'
# Directory of images to run detection on
#IMAGE_DIR = os.path.join("E:\DataSet\CDWdevkit\CDW2020\\Hyperspec_test14\\")
#IMAGE_DIR = os.path.join("E:\DataSet\CDWdevkit\CDW2020\\Image_test\\")
IMAGE_DIR = "E:/20201203/val/image"



class InferenceConfig(cdw.CDWConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.7

config = InferenceConfig()
config.display()

#%% md

## Create Model and Load Trained Weights

#%%

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=CDW_MODEL_PATH, config=config)

# Load weights trained on MS-COCO
model.load_weights(CDW_MODEL_PATH, by_name=True)#, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])
#model.keras_model.save('mask_rcnn.h5')

#%%

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

class_names = ['BG','concrete','gray','red','wood','plaster','plastic','ceramic','carton']


# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
img_name = os.path.join(IMAGE_DIR, random.choice(file_names))
image = skimage.io.imread(img_name)
height_name = img_name.replace('/image', '/Height')
height = skimage.io.imread(height_name)
height = np.expand_dims(height, axis=-1)
image = np.concatenate([image, height], axis=-1)


print(image.shape)
# image=np.swapaxes(image,0,1)
# image=np.swapaxes(image,1,2)
# image=image*255
# image=image.astype(np.uint8)

# Run detection
import time
t1=time.clock()
results = model.detect([image], verbose=1)
print("time:",time.clock()-t1)
# Visualize results
r = results[0][0]


# print(image[:,:,:3])
# print(r['rois'])
# print(r['masks'], r['class_ids'])
# print(class_names)
# visualize.display_instances(image[:,:,:3], r['rois'], r['masks'], r['class_ids'],
                            # class_names, r['scores'])
# print( r['class_ids'])


# Validation dataset
dataset_root='E:/20201203'
dataset= cdw.CDWDataset()
#dataset.load_cdw(dataset_root, json_dir="Annotation",img_dir='Hyperspec_',hyper=True)
#dataset.load_cdw(dataset_root, json_dir="Annotation",img_dir=None,hyper=False)
dataset.load_cdw(dataset_root, "val")
dataset.prepare()
print('val data prepared')

import imgaug
# Compute VOC-style Average Precision
def compute_batch_ap(image_ids):
    AP50s = []
    AP75s = []
    count = 0
    for image_id in image_ids:
        # Load image
        print(count)
        count+=1
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, config,
                                   image_id,
                                   use_mini_mask=False,
                                   # augmentation=imgaug.augmenters.Sometimes(0.5, [
                                   #  imgaug.augmenters.Fliplr(0.5),
                                   #  imgaug.augmenters.Flipud(0.5),
                                   #  imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                                   #  ])
                                   )
        # Run object detection
        results = model.detect([image], verbose=0)
        # Compute AP
        r = results[0][0]
        # visualize.display_instances(image[:,:,:3], r['rois'], r['masks'], r['class_ids'],
        #                     class_names, r['scores'])
        AP, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r['rois'], r['class_ids'], r['scores'], r['masks'], 0.75)
        #confuse[gt_class_id[:],r['class_ids'][:]]+=1
        AP75s.append(AP)

        AP, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r['rois'], r['class_ids'], r['scores'], r['masks'], 0.5)
        #confuse[gt_class_id[:],r['class_ids'][:]]+=1
        AP50s.append(AP)
    print("Test nums", len(AP50s))
    return AP50s, AP75s
AP50s, AP75s = compute_batch_ap(dataset.image_ids)

#%%

image, image_meta, gt_class_id, gt_bbox, gt_mask = \
    modellib.load_image_gt(dataset, config,
                           dataset.image_ids[1], use_mini_mask=False)
results = model.detect([image], verbose=0)
t1=time.clock()
utils.compute_overlaps_masks(gt_mask,results[0][0]['masks'])
print("Calculate mask iou time cost: ",time.clock()-t1)

#%%

a=dataset.image_ids
mAp=np.nanmean(AP50s)
print("mAP @ IoU=50: ", mAp)
mAp=np.nanmean(AP75s)
print("mAP @ IoU=75: ", mAp)

#%%

# images=[]
# mask_gt=np.empty(shape=[960,1024,0])
# bbox_gt=np.empty(shape=[0,4])
# class_gt=np.empty(shape=[1,0])
#
# bbox_pred=np.empty(shape=[0,4])
# mask_pred=np.empty(shape=[960,1024,0])
# class_pred=np.empty(shape=[1,0])
# score_pred=np.empty(shape=[1,0])
# for image_id in dataset.image_ids:
#     image = dataset.load_image(image_id)
#     mask, class_ids = dataset.load_mask(image_id)
#     bbox = utils.extract_bboxes(mask)
#     bbox_gt=np.append(bbox_gt,bbox,0)
#     images.append(image)
#     mask_gt=np.append(mask_gt,mask,2)
#     class_gt=np.append(class_gt,class_ids)
#
#     r=model.detect([image], verbose=1)[0]
#     bbox_pred=np.append(bbox_pred,r['rois'],0)
#     mask_pred=np.append(mask_pred,r['masks'],2)
#     class_pred=np.append(class_pred,r['class_ids'])
#     score_pred=np.append(score_pred,r['scores'])
#     visualize.display_instances(image[:,:,:3], r['rois'], r['masks'], r['class_ids'],
#                             class_names, r['scores'])
#

#%%

# from mrcnn import utils
# #for i in range(len(bbox_gt)):
#
# mAP, precisions, recalls, overlaps=utils.compute_ap(bbox_gt,class_gt,mask_gt,
#                                               bbox_pred,class_pred,score_pred,
#                                             mask_pred,iou_threshold=0.5)
# for i in range(1,5):
#     dataset_train_dir="Hyperspec\\train_%i"%i
#     print(dataset_train_dir)
