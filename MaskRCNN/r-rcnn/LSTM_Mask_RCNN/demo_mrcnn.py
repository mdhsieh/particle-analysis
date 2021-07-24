# new script using nucleus.py from https://github.com/matterport/Mask_RCNN/tree/master/samples/nucleus
# Do detection of cells on images in demo_video_nucleus/frames
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
ROOT_DIR = os.getcwd()
print("root dir:", ROOT_DIR)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

# to read image only
import skimage.io

# Directory to save logs and trained model
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Dataset directory
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

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference",
                              model_dir=LOGS_DIR,
                              config=config)

# Path to a specific weights file
weights_path = "kaggle_bowl.h5"

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

# loop through all images with extension in dataset dir and perform detection
for filename in os.listdir(DATASET_DIR):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_name = filename
        img_path = os.path.join(DATASET_DIR, filename)
        # load image code from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/utils.py
        # Load image
        image = skimage.io.imread(img_path)
        # remove alpha channel to get 3 dim instead of 4
        image = image[:,:,:3]
        print("removed alpha channel! new image shape:", image.shape)
        
        # regular detection function instead of model.detect_molded()
        print("object detection using detect()")
        print("image name: " + img_name)
        
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

print("Done. Results saved in " + RESULT_DIR)