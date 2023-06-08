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




MODEL_DIR = os.path.join(ROOT_DIR, "logs")
os.environ['CUDA_VISIBLE_DEVICES']='1'
CDW_MODEL_PATH = 'D:/Project_py/maskrcnn_rgbh/logs/cdw20200619T0916/mask_rcnn_cdw_0040.h5'
IMAGE_DIR = "E:/CDW/val/image"



class InferenceConfig(cdw.CDWConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.7

config = InferenceConfig()
config.display()


model = modellib.MaskRCNN(mode="inference", model_dir=CDW_MODEL_PATH, config=config)

model.load_weights(CDW_MODEL_PATH, by_name=True)

class_names = ['BG','wood','brick','concrete','rubber']


# Load a random image from the images folder
# file_names = next(os.walk(IMAGE_DIR))[2]
# img_name = os.path.join(IMAGE_DIR, random.choice(file_names))
# image = skimage.io.imread(img_name)
# height_name = img_name.replace('/image', '/Height')
# height = skimage.io.imread(height_name)
# height = np.expand_dims(height, axis=-1)
# image = np.concatenate([image, height], axis=-1)
#
# import time
# t1=time.clock()
# results = model.detect([image], verbose=1)
# print("time:",time.clock()-t1)
# # Visualize results
# r = results[0]
# visualize.display_instances(image[:,:,:3], r['rois'], r['masks'], r['class_ids'],
#                             class_names, r['scores'])
# print( r['class_ids'])


# Validation dataset
dataset_root='E:/CDW/'
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
    ious = []
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
        results, rpn_rois, _ = model.detect([image], verbose=0)

        # Compute AP
        r = results[0]

        # print(gt_bbox.shape)
        # print(gt_bbox)
        # print(rpn_rois[0])
        iou = utils.compute_overlaps(gt_bbox, rpn_rois[0]*640)

        iou = np.max(iou, -1)
        # print(iou)

        # delete_index = iou > 0.5
        # # print(delete_index)
        # iou = iou[delete_index]

        if len(ious):
            ious = np.concatenate([ious, iou])
        else:
            ious = iou
        # print(iou)

        # print(np.max(ious, -1))
        # visualize.display_instances(image[:,:,:3], r['rois'], r['masks'], r['class_ids'],
        #                     class_names, r['scores'])
        # AP, precisions, recalls, overlaps = \
        #     utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
        #                      r['rois'], r['class_ids'], r['scores'], r['masks'], 0.75)
        # #confuse[gt_class_id[:],r['class_ids'][:]]+=1
        # AP75s.append(AP)
        #
        # AP, precisions, recalls, overlaps = \
        #     utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
        #                      r['rois'], r['class_ids'], r['scores'], r['masks'], 0.5)
        # #confuse[gt_class_id[:],r['class_ids'][:]]+=1
        # AP50s.append(AP)
    print("测试数量",len(AP50s))
    return ious
ious = compute_batch_ap(dataset.image_ids)

#%%
print(ious)
print(np.mean(ious))

delete_index = ious > 0.6
# print(delete_index)
iou = ious[delete_index]

print('Recall iou>0.6: ', len(iou) / len(ious))

delete_index = ious > 0.7
# print(delete_index)
iou = ious[delete_index]

print('Recall iou>0.6: ', len(iou) / len(ious))

delete_index = ious > 0.8
# print(delete_index)
iou = ious[delete_index]

print('Recall iou>0.6: ', len(iou) / len(ious))

delete_index = ious > 0.9
# print(delete_index)
iou = ious[delete_index]

print('Recall iou>0.6: ', len(iou) / len(ious))







#%%


