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
	# Set batch size to 1 since we'll be running inference on
	# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	IMAGE_MIN_DIM = 480
	IMAGE_MAX_DIM = 640

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
