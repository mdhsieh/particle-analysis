"""
Recurrent Region CNN
The training implementation of R-RCNN.

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
import argparse

import utils.utils as utils
import utils.visualize as visualize
import detection_module.coco as coco
import detection_module.detect_model as detectlib
import tracking_module.track_config as trackconfig
import tracking_module.track_model as tracklib

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
coco_class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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

train_names = ['BlurBody_train', 'BlurCar1_train', 'BlurCar3_train',  
				'CarDark_train', 'CarScale_train', 'Couple_train', 'David3_train', 
				'Dog_train', 'Girl2_train', 'Gym_train', 'Human2_train', 'Human3_train', 
				'Human6_train', 'Human7_train', 'Human8_train', 'Human9_train', 'Jump_train', 
				'Singer1_train', 'Singer2_train', 'Skating1_train', 'Walking2_train', 'Woman_train',
				'Basketball_train', 'BlurCar4_train', 'Bolt2_train', 'Car24_train', 'Coke_train', 'Crossing_train', 
				'Human4_train', 'Human5_train', 'Jogging1_train', 'Jogging2_train', 'Liquor_train',
				'RedTeam_train', 'Skating2_train', 'Skiing_train', 'Subway_train', 'Walking_train']

val_names = ['BlurBody_val', 'BlurCar1_val', 'BlurCar3_val', 
				'CarDark_val', 'CarScale_val', 'Couple_val', 'David3_val',
				'Dog_val', 'Girl2_val', 'Gym_val', 'Human2_val', 'Human3_val', 'Human6_val', 
				'Human7_val', 'Human8_val', 'Human9_val', 'Jump_val', 'Singer1_val', 
				'Singer2_val', 'Skating1_val', 'Walking2_val', 'Woman_val',
				'Basketball_val', 'BlurCar4_val', 'Bolt2_val', 'Car24_val', 'Coke_val', 'Crossing_val', 
				'Human4_val', 'Human5_val', 'Jogging1_val', 'Jogging2_val', 'Liquor_val',
				'RedTeam_val', 'Skating2_val', 'Skiing_val', 'Subway_val', 'Walking_val']

argparser = argparse.ArgumentParser(
    description='Training of Tracking Module')

argparser.add_argument(
    '-w',
    dest='weight',
    help='name of the pre-trained weights')

args = argparser.parse_args()

ROOT_DIR = os.getcwd()
# Directory to pre-processed Mask RCNN data
DATA_DIR = os.path.join(ROOT_DIR, "Detection_DATA")
# Directory to dataset
DATASET_DIR = os.path.join(ROOT_DIR, "OTB")
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Weights filename for saving (without .h5)
WEIGHTS_NAME = "OTB_T4B32"


class LSTMTrainingConfig(trackconfig.Config):
	NAME = "OTB"
	TIME_STEPS = 4
	STEPS_PER_EPOCH = 1000
	EPOCHS = 50
	LEARNING_RATE = 1e-5
	

############################################################
#  LSTM training
############################################################
trackconfig = LSTMTrainingConfig()
trackconfig.display()

lstmModel = tracklib.LSTM(mode="training", config=trackconfig, 
	model_dir=MODEL_DIR, data_dir=DATA_DIR, dataset_dir=DATASET_DIR, w_name=WEIGHTS_NAME)


init_with = args.weight 

if init_with == "None":
	# Train from scratch
	lstmModel.train(train_dataset=train_names, val_dataset=val_names)

else:
	# Path to pre-trained LSTM weights file
	OTB_MODEL_PATH = os.path.join(ROOT_DIR, init_with)
	# Load pre-trained weights
	lstmModel.keras_lstm_model.load_weights(OTB_MODEL_PATH)
	lstmModel.train(train_dataset=train_names, val_dataset=val_names)
