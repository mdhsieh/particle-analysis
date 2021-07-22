"""
Mask R-CNN
Mask R-CNN demo.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import time

import utils.utils as utils
import utils.visualize as visualize
import coco
import detection_module.detect_model as detectlib

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
coco_class_names = ['BG', 'cell']

# Root directory of the project
ROOT_DIR = os.getcwd()
# Path to trained weights file
# Download this file and place in the root of your 
# project (See README file for details)
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "kaggle_bowl.h5")
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "demo_video_nucleus", "frames")
# Directory of MRCNN for examined
RESULT_DIR = os.path.join(ROOT_DIR, "mrcnn_result")
if not os.path.exists(RESULT_DIR):
	os.makedirs(RESULT_DIR)


class InferenceConfig(coco.CocoConfig):
# 	# Set batch size to 1 since we'll be running inference on
# 	# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
# 	GPU_COUNT = 1
# 	IMAGES_PER_GPU = 1
# 	IMAGE_MIN_DIM = 480
# 	IMAGE_MAX_DIM = 640
    # Use NucleusInferenceConfig from https://github.com/matterport/Mask_RCNN/tree/master/samples/nucleus/nucleus.py
	# Configuration for training on the nucleus segmentation dataset.
    # Give the configuration a recognizable name
    NAME = "nucleus"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + nucleus

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    # DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = detectlib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# Load weights trained on MS-COCO
# model.load_weights(COCO_MODEL_PATH, by_name=True)
model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
file_names.sort()

# Initial total run time
total_mrcnn_time = 0
# The number of frames
frame_number = len(file_names)

for img_name in file_names:
	print("Image name: %s" % (img_name))
	# Time start
	mrcnn_start = time.time()
	image = skimage.io.imread(os.path.join(IMAGE_DIR, img_name))
	#image = skimage.io.imread('/home/ubuntueric/darknet_1204/ITRI_test/00094_PT.jpg')

	# Run detection
	results = model.detect([image], verbose=1)
	r = results[0]
	# Time end
	mrcnn_end = time.time()
	total_mrcnn_time = total_mrcnn_time + (mrcnn_end - mrcnn_start)

	# Visualize results
	visualize.save_image(image, img_name, r['rois'], r['masks'],
		r['class_ids'], r['scores'], coco_class_names,
		filter_classs_names=None, scores_thresh=0.1, 
		save_dir=RESULT_DIR, mode=1)

fps = frame_number / total_mrcnn_time
print("\n# ------------------------------------ #")
print("# Processed Frames: %d" % frame_number)
print("# Cost Time: %.3f" % total_mrcnn_time)
print("# FPS: %.1f" % fps)
print("# ------------------------------------ #\n")

# new script using nucleus.py from https://github.com/matterport/Mask_RCNN/tree/master/samples/nucleus
'''
import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# go to samples/nucleus folder in LSTM_Mask_RCNN repo
sys.path.append("samples/nucleus")
# import nucleus is only available in samples/nucleus directory
import nucleus

# Root directory of the project
# ROOT_DIR = os.path.abspath("../../")
ROOT_DIR = os.getcwd()
print("root dir:", ROOT_DIR)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

# %matplotlib inline 

# Directory to save logs and trained model
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Dataset directory
# DATASET_DIR = os.path.join(ROOT_DIR, "single-particle-dataset/nucleus")
DATASET_DIR = os.path.join(ROOT_DIR, "demo_video_nucleus/frames")

# Inference Configuration
config = nucleus.NucleusInferenceConfig()
config.display()

# Device to load the neural network on.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# Only inference mode is supported right now
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    fig, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    fig.tight_layout()
    return ax

# Load validation dataset
dataset = nucleus.NucleusDataset()
dataset.load_nucleus(DATASET_DIR, "stage1_test")

dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference",
                              model_dir=LOGS_DIR,
                              config=config)

# Path to a specific weights file
# weights_path = "../../kaggle_bowl.h5"
weights_path = "kaggle_bowl.h5"

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

image_id = random.choice(dataset.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset.image_reference(image_id)))
print("Original image shape: ", modellib.parse_image_meta(image_meta[np.newaxis,...])["original_image_shape"][0])

# regular detection function
print("object detection using detect()")
img_name = str(image_id) + ".png" # ".jpg"
import utils.utils as utils
import utils.visualize as visualize
coco_class_names = ['BG', 'cell']
RESULT_DIR = os.path.join(ROOT_DIR, "mrcnn_result")
regular_results = model.detect([image], verbose=1)
reg_r = regular_results[0]
# Visualize results
visualize.save_image(image, img_name, reg_r['rois'], reg_r['masks'],
  reg_r['class_ids'], reg_r['scores'], coco_class_names,
  filter_classs_names=None, scores_thresh=0.1, 
  save_dir=RESULT_DIR, mode=1)

print("Done")
'''