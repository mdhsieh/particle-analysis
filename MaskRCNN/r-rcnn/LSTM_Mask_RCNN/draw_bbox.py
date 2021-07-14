"""
Recurrent Region CNN
Draw the results of R-RCNN, Mask R-CNN and Grondtruth.

Copyright (c) 2018 Chen-En Chung
Licensed under the MIT License (see LICENSE for details)
Written by Chen-En, Chung
"""

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import tracking_module.track_config as trackconfig
import utils.utils as utils
import utils.visualize as visualize

val_names = ['BlurBody_val', 'BlurCar1_val', 'BlurCar3_val', 
				'CarDark_val', 'CarScale_val', 'Couple_val', 'David3_val',
				'Dog_val', 'Girl2_val', 'Gym_val', 'Human2_val', 'Human3_val', 'Human6_val', 
				'Human7_val', 'Human8_val', 'Human9_val', 'Jump_val', 'Singer1_val', 
				'Singer2_val', 'Skating1_val', 'Walking2_val', 'Woman_val',
				'Basketball_val', 'BlurCar4_val', 'Bolt2_val', 'Car24_val', 'Coke_val', 'Crossing_val', 
				'Human4_val', 'Human5_val', 'Jogging1_val', 'Jogging2_val', 'Liquor_val',
				'RedTeam_val', 'Skating2_val', 'Subway_val', 'Walking_val']

argparser = argparse.ArgumentParser(
    description='Visualization of results')

argparser.add_argument(
    '-w',
    dest='weight',
    help='name of the trained weights')

args = argparser.parse_args()

W_NAME = args.weight[:-3]
ROOT_DIR = os.getcwd()
LSTM_DIR = os.path.join(ROOT_DIR, "RRCNN_results", W_NAME)
# Directory to dataset
DATASET_DIR = os.path.join(ROOT_DIR, "OTB")
MRCNN_DIR = os.path.join(ROOT_DIR, "Detection_DATA")

class LSTMInferenceConfig(trackconfig.Config):
	TIME_STEPS = 4

config = LSTMInferenceConfig()

for folder in val_names:
	# Directory to save image results
	mrcnn_result_dir = os.path.join(LSTM_DIR, "val_image", folder)
	if not os.path.exists(mrcnn_result_dir):
		os.makedirs(mrcnn_result_dir)

	# Directory of images to run detection on
	img_dir = os.path.join(DATASET_DIR, folder, "img")

	# Directory of Annotations
	ann_name = os.path.join(MRCNN_DIR, "bbox_GT", folder+".npy")
	ann_bbox = np.load(ann_name)
	# Directory of MRCNN results
	mrcnn_bbox_name = os.path.join(MRCNN_DIR, "true_bbox", folder+".npy")
	mrcnn_bbox = np.load(mrcnn_bbox_name)
	# Directory of LSTM resutls
	lstm_bbox_name = os.path.join(LSTM_DIR, "bbox", folder+".npy")
	lstm_bbox = np.load(lstm_bbox_name)
	
	# Load images from the image folder
	file_names = next(os.walk(img_dir))[2]
	file_names.sort()

	for idx, img_name in enumerate(file_names):
		if idx >= config.TIME_STEPS:
			# Read image
			image = skimage.io.imread(os.path.join(img_dir, img_name)) 
			# Get H, W of image for normalization
			H, W = image.shape[:2]
			print("Image name: {}/{}".format(folder, img_name))

			############################################################
			#  Ground Truth
			############################################################
			ann_box = utils.bbox_toyxyx(ann_bbox[idx, :])
			ann_box = utils.bbox_denormalization(ann_box, H=H, W=W)
			############################################################
			#  MRCNN
			############################################################
			mrcnn_box = utils.bbox_toyxyx(mrcnn_bbox[idx, :])
			mrcnn_box = utils.bbox_denormalization(mrcnn_box, H=H, W=W)
			############################################################
			#  LSTM
			############################################################
			lstm_box = utils.bbox_toyxyx(lstm_bbox[idx, :])
			lstm_box = utils.bbox_denormalization(lstm_box, H=H, W=W)


			# Combine bboxes by the order of GT & MRCNN & LSTM 
			boxes = np.vstack((ann_box, mrcnn_box, lstm_box))
			# Draw & Save
			visualize.save_tracking_bbox(image, img_name, boxes, 
					save_dir=mrcnn_result_dir, mode=0)

		