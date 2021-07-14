"""
Recurrent Region CNN
R-RCNN demo.

Copyright (c) 2018 Chen-En Chung
Licensed under the MIT License (see LICENSE for details)
Written by Chen-En, Chung
"""

import os
import sys
import random
import math
import time
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import utils.utils as utils
import utils.visualize as visualize
import coco
import detection_module.detect_model as detectlib
import tracking_module.track_config as trackconfig
import tracking_module.track_model as tracklib



# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
coco_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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

# Root directory of the project
ROOT_DIR = os.getcwd()
# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Path to LSTM Model weights
LSTM_MODEL_PATH = os.path.join(ROOT_DIR, "RRCNN_OTB+.h5")
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Directory to root of dataset
DATASET_DIR = os.path.join(ROOT_DIR, "OTB")
# Directory of video frames to run tracking on
IMAGE_DIR = os.path.join(ROOT_DIR, "demo_video", "frames")
# Path to 1st frames annotation
ANNO_PATH = os.path.join(ROOT_DIR, "demo_video", "annotation.txt")
# Path to 1st frames class id
CLASSID_PATH = os.path.join(ROOT_DIR, "demo_video", "coco_id.txt")
# Directory to save image results
RESULT_DIR = os.path.join(ROOT_DIR, "rrcnn_result")
if not os.path.exists(RESULT_DIR):
	os.makedirs(RESULT_DIR)

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640

class LSTMInferenceConfig(trackconfig.Config):
	# One frame per batch for inference
	BATCH_SIZE = 1
	TIME_STEPS = 4


def demo():
	# Load images from the images folder
	img_names = next(os.walk(IMAGE_DIR))[2]
	img_names.sort()
	# Load target class id
	target_class = int(np.genfromtxt(CLASSID_PATH))
	# Initial LSTM input
	X_bbox = np.empty((trackconfig.BATCH_SIZE, trackconfig.TIME_STEPS, trackconfig.MRCNNBBOX_SIZE))
	X_feat = np.empty((trackconfig.BATCH_SIZE, trackconfig.TIME_STEPS, trackconfig.FEATURE_SIZE))
	# Initial total run time
	total_rrcnn_time = 0
	# The number of frames
	frame_number = len(img_names)

	# Delay tracking start
	for idx, img_name in enumerate(img_names):
		# Time start
		rrcnn_start = time.time()
		# Read image
		image = skimage.io.imread(os.path.join(IMAGE_DIR, img_name))
		# Get H, W of image for normalization
		H, W = image.shape[:2]
		print("Image name: %s" % (img_name))
		# Using annotation for 1st frame  
		if idx == 0:
			anno_bbox = np.genfromtxt(ANNO_PATH, delimiter=',') # [y, x, w, h]
			# Keep as last frame info.
			last_bbox = utils.bbox_toyxyx(bbox=anno_bbox) # [y1, x1, y2, x2]
			last_bbox = utils.bbox_denormalization(bbox=last_bbox, H=H, W=W) # [0, H]
			last_feat = np.zeros((trackconfig.FEATURE_SIZE, ))

		# MRCNN detects until reach TIME_STEPS
		elif idx <= trackconfig.TIME_STEPS:		
			############################################################
			#  Mask RCNN detection
			############################################################
			result = model.detect([image], verbose=1)
			r = result[0]	# r['rois']: [N, (y1, x1, y2, x2)] detection bounding boxes
							# r['class_ids']: [N] int class IDs
							# r['scores']: [N] float probability scores for the class IDs
							# r['masks']: [H, W, N] instance binary masks
			# Extract feature map 
			feature_map = model.run_graph([image], [
				('squeeze', model.keras_model.get_layer('pool_squeeze').output),
			])
			feature = feature_map['squeeze'][0][r['roi_index'].astype(np.int32)]

			# Use last results to filter target results from all Mask RCNN results
			filter_bbox, filter_feat, mrcnn_bbox = \
				utils.targetbbox_filter(
					image_name=img_name, last_bbox=last_bbox,
					last_feat=last_feat, target_class=target_class, 
					mrcnn_bbox=r["rois"], mrcnn_feat=feature, mrcnn_class=r['class_ids'], 
					thresh=trackconfig.THRESHOLD)

			# Keep last frame's info.
			last_bbox = filter_bbox # [y1, x1, y2, x2]
			last_feat = filter_feat
			# Keep LSTM input
			filter_bbox = utils.bbox_normalization(bbox=filter_bbox, H=H, W=W) # [0, 1]
			X_bbox[0, idx-1] = utils.bbox_toyxhw(bbox=filter_bbox) # [y, x, h, w]
			X_feat[0, idx-1] = filter_feat

			# Start Tracking when reach TIME_STEPS
			if idx == trackconfig.TIME_STEPS:
				############################################################
				#  LSTM Tracking
				############################################################
				lstm_result = lstmModel.detect(lstmInput=[X_bbox, X_feat])
				lstm_bbox = lstm_result[0] # [y, x, h, w][0, 1]
				# Update last frame's info. from LSTM
				last_lstm = lstm_bbox
				last_bbox = utils.bbox_toyxyx(bbox=lstm_bbox) # [y1, x1, y2, x2]
				last_bbox = utils.bbox_denormalization(bbox=last_bbox, H=H, W=W) # [0, H]

		# After TIME_STEPS
		else:
			if trackconfig.TIME_STEPS > 2:
				X_bbox[0, :trackconfig.TIME_STEPS-2] = X_bbox[0, 1:trackconfig.TIME_STEPS-1]
			# Using LSTM result to replace MRCNN result
			X_bbox[0, trackconfig.TIME_STEPS-2] = last_lstm
			############################################################
			#  Mask RCNN detection
			############################################################
			result = model.detect([image], verbose=1)
			r = result[0]	# r['rois']: [N, (y1, x1, y2, x2)] detection bounding boxes
							# r['class_ids']: [N] int class IDs
							# r['scores']: [N] float probability scores for the class IDs
							# r['masks']: [H, W, N] instance binary masks
			# Extract feature map 
			feature_map = model.run_graph([image], [
				('squeeze', model.keras_model.get_layer('pool_squeeze').output),
			])
			feature = feature_map['squeeze'][0][r['roi_index'].astype(np.int32)]

			# Use last results to filter target results from all Mask RCNN results
			filter_bbox, filter_feat, mrcnn_bbox = \
				utils.targetbbox_filter(
					image_name=img_name, last_bbox=last_bbox,
					last_feat=last_feat, target_class=target_class, 
					mrcnn_bbox=r["rois"], mrcnn_feat=feature, mrcnn_class=r['class_ids'], 
					thresh=trackconfig.THRESHOLD)

			# Using MRCNN result for the last time step
			filter_bbox = utils.bbox_normalization(bbox=filter_bbox, H=H, W=W) # [0, 1
			X_bbox[0, trackconfig.TIME_STEPS-1] = utils.bbox_toyxhw(bbox=filter_bbox) # [y, x, h, w]

			# Keep last frame's feature
			last_feat = filter_feat
			# Update feature for LSTM input
			X_feat[0, :trackconfig.TIME_STEPS-1] = X_feat[0, 1:]
			X_feat[0, trackconfig.TIME_STEPS-1] = filter_feat

			############################################################
			#  LSTM Tracking
			############################################################
			lstm_result = lstmModel.detect(lstmInput=[X_bbox, X_feat])
			lstm_bbox = lstm_result[0] # [y, x, h, w][0, 1]
			# Update last frame's info. from LSTM
			last_lstm = lstm_bbox
			last_bbox = utils.bbox_toyxyx(bbox=lstm_bbox) # [y1, x1, y2, x2]
			last_bbox = utils.bbox_denormalization(bbox=last_bbox, H=H, W=W) # [0, H]

		# Time end
		rrcnn_end = time.time()
		total_rrcnn_time = total_rrcnn_time + (rrcnn_end - rrcnn_start)

		############################################################
		#  Draw result & save
		############################################################
		if idx >= trackconfig.TIME_STEPS:
			lstm_bbox = utils.bbox_toyxyx(bbox=lstm_bbox) # [y1, x1, y2, x2]
			lstm_bbox = utils.bbox_denormalization(bbox=lstm_bbox, H=H, W=W) # [0, H]
			
			# Red: RRCNN bbox
			# Green: MRCNN bbox
			boxes = np.vstack((lstm_bbox, mrcnn_bbox))
			visualize.save_tracking_bbox(image, img_name, boxes, save_dir=RESULT_DIR, mode=0)

	fps = frame_number / total_rrcnn_time
	print("\n# ------------------------------------ #")
	print("# Build LSTM Model Time: %.3f" % buildLSTM_time)
	print("# Processed Frames: %d" % frame_number)
	print("# Cost Time: %.3f" % total_rrcnn_time)
	print("# FPS: %.1f" % fps)
	print("# ------------------------------------ #\n")


if __name__ == '__main__':
	############################################################
	#  Mask RCNN setting
	############################################################
	# Customize model config
	config = InferenceConfig()
	config.display()
	# Create model object in inference mode.
	model = detectlib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
	# Load weights trained on MS-COCO
	model.load_weights(COCO_MODEL_PATH, by_name=True)

	############################################################
	#  LSTM setting
	############################################################
	buildLSTM_start = time.time()
	trackconfig = LSTMInferenceConfig()
	trackconfig.display()

	lstmModel = tracklib.LSTM(mode="inference", config=trackconfig, model_dir=MODEL_DIR)
	lstmModel.keras_lstm_model.load_weights(LSTM_MODEL_PATH)
	buildLSTM_time = time.time() - buildLSTM_start

	demo()
