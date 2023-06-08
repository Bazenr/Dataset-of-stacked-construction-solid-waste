"""
Mask R-CNN
Train on your dataset (CDW is my research Construction and Demolition Waste).

Revised according to the work:
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 cdw.py train --dataset=/path/to/cdw/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 cdw.py train --dataset=/path/to/cdw/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 cdw.py train --dataset=/path/to/cdw/dataset --weights=imagenet

"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import glob
import cv2 as cv
import imgaug


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
# from mrcnn.config import Config
from samples.cdw.config_CDW_RGBH import Config
from mrcnn import model as modellib, utils
from samples.cdw.warmup import WarmUpCosineDecayScheduler
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class CDWConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "cdw"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # Background + wood, brick, concrete and rubber

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 500
    VALIDATION_STEPS = 174
    # Skip detections with < 50% confidence
    DETECTION_MIN_CONFIDENCE = 0.7   # 0.5 - 0.9


############################################################
#  Dataset
############################################################

class CDWDataset(utils.Dataset):

    def load_cdw(self, dataset_dir, subset):
        """Load a subset of the CDW dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("cdw", 1, "wood")
        self.add_class("cdw", 2, "brick")
        self.add_class("cdw", 3, "concrete")
        self.add_class("cdw", 4, "rubber")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        
        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {'class':class_name},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations_json_path = os.path.join(dataset_dir, 'json')
        annotations_json_path = os.path.join(annotations_json_path, '*.json')
        annotations = glob.glob(annotations_json_path)   #[1:10]  #[0:10]

        images = [a.replace('\\json', '\\image') for a in annotations]
        images = [a.replace('.json', '.png') for a in images]
        # print(annotations)
        # print(len(annotations))

        # print(images)
        # print(len(images))


        count = 1
        for image_path,annot in zip(images, annotations):

            print('loading '+ subset + ' data' +'{: d} / {: d}'.format(count, len(images)))
            count += 1
            image = skimage.io.imread(image_path)
            file_name = os.path.basename(image_path)
            # print(file_name)
            height, width = image.shape[:2]
            with open(annot, 'r') as f:
                json_text = json.load(f)
            shapes = json_text.get('shapes', None)
            polygons = []
            class_names = []
            if not len(shapes):
                continue
            for mark in shapes:
                class_name = mark.get('label')
                class_names.append(class_name)
                polygon = mark.get('points')
                #print(polygon)
                polygons.append(polygon)

            self.add_image(
                "cdw",
                image_id=file_name,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                cdw=class_names)


    def load_image(self, image_id):
         # Load image

        path = self.image_info[image_id]['path']
        image = skimage.io.imread(path)
        # height = skimage.io.imread(path.replace('\\image', '\\height'))
        # height = np.expand_dims(height, axis=-1)
        # res = np.concatenate([image, height], axis = -1)

        # # If grayscale. Convert to RGB for consistency.
        # if image.ndim != 3:
        #     image = skimage.color.gray2rgb(image)
        # # If has an alpha channel, remove it for consistency
        # if image.shape[-1] == 4:
        #     image = image[..., :3]
        return image  # res


    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a cdw dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "cdw":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        #print('image info',info)
        cdws = info['cdw']
        #print('cdws',cdws)
        # mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
        #                 dtype=np.uint8)
        masks = []
        # print(mask.shape)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            # rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            # mask[rr, cc, i] = 1
            mask = np.zeros([info['height'], info['width']], dtype = np.uint8)
            polygon = np.array(p,dtype=np.int)
            # print(mask[:,:,i].shape)
            cv.fillPoly(mask, [polygon], 1)
            masks.append(mask)

        masks = np.stack(masks, axis = -1)
        # Return mask, and array of class IDs of each instance.
        class_ids = np.array([self.class_names.index(s) for s in cdws])
        return masks,class_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cdw":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CDWDataset()
    dataset_train.load_cdw(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CDWDataset()
    dataset_val.load_cdw(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.

    augmentation = imgaug.augmenters.Sometimes(0.5,
               [imgaug.augmenters.Dropout(p=(0, 0.1)),  # 将0-10%的像素置零
                imgaug.augmenters.Multiply(),  # 将像素随机×（0.8-1.2）
                imgaug.augmenters.Fliplr(0.5),  # 水平翻转 
                imgaug.augmenters.Flipud(0.5),  # 竖直反转
                imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0)),  # 高斯平滑
                imgaug.augmenters.AddElementwise((-5, 10), per_channel=0.5)
                ])

    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=3,
    #             layers=r"(conv1)")

    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=30,
    #             layers=r"(conv1)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=30,
    #             layers='all')

    # # Training - Stage 2
    # # # Finetune layers from ResNet stage 4 and up
    # print("Fine tune Resnet stage 4 and up")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=60,
    #             layers='heads')

    # # Training - Stage 3
    # # Fine tune all layers
    # print("Fine tune all layers")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE / 10,
    #             epochs=90,
    #             layers='all')

    # print("Fine tune all layers")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE / 40,
    #             epochs=120,
    #             layers='all')

    # print("Training network heads")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=20,
    #             layers='heads')
    warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=config.learning_rate_base,
                                                total_steps=config.total_steps,
                                                warmup_learning_rate=1e-04,
                                                warmup_steps=config.warmup_steps,
                                                hold_base_rate_steps=5,
                                                verbose=1)

    # Train the model, iterating on the data in batches of 32 samples
    # model.fit(data, one_hot_labels, epochs=epochs, batch_size=config.BATCH_SIZE,
    #             verbose=0, callbacks=[warm_up_lr])
    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=config.epochs,
                layers='all',
                augmentation=augmentation,
                custom_callbacks=[warm_up_lr])



############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("--command",default="train",required=False,
                        metavar="<command>",
                        help="train")
    parser.add_argument('--dataset', required=False,
                        default='E:/20201203/',
                        metavar="/path/to/cdw/dataset/",
                        help='Directory of the CDW dataset')
    parser.add_argument('--weights', required=False,
                        # default='D:/Project_py/maskrcnn_my/logs/cdw20200611T1559/mask_rcnn_cdw_0020.h5',
                        default='D:/Project_py/maskrcnn_rgbh/logs/cdw20201228T1916/mask_rcnn_cdw_0082.h5',
                        #default='D:/Project_py/maskrcnn_rgbh/mask_rcnn_coco.h5',
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CDWConfig()
    else:
        class InferenceConfig(CDWConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)

    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask", "conv1"])
    # else:
    #     model.load_weights(weights_path, by_name=True)#, exclude=["conv1","mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])
        
        # model.load_weights(weights_path, by_name=True)
    # Train or evaluate
    if args.command == "train":
        train(model)

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
