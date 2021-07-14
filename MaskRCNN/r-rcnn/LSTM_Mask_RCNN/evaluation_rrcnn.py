"""
Recurrent Region CNN
The evaluation implemenetation of R-RCNN.

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

import detection_module.detect_model as detectlib
import tracking_module.track_config as trackconfig
import tracking_module.track_model as tracklib
import utils.utils as utils
import utils.visualize as visualize


val_names = ['BlurBody_val', 'BlurCar1_val', 'BlurCar3_val', 
				'CarDark_val', 'CarScale_val', 'Couple_val', 'David3_val',
				'Dog_val', 'Girl2_val', 'Gym_val', 'Human2_val', 'Human3_val', 'Human6_val', 
				'Human7_val', 'Human8_val', 'Human9_val', 'Jump_val', 'Singer1_val', 
				'Singer2_val', 'Skating1_val', 'Walking2_val', 'Woman_val',
				'Basketball_val', 'BlurCar4_val', 'Bolt2_val', 'Car24_val', 'Coke_val', 'Crossing_val', 
				'Human4_val', 'Human5_val', 'Jogging1_val', 'Jogging2_val', 'Liquor_val',
				'RedTeam_val', 'Skating2_val', 'Skiing_val', 'Subway_val', 'Walking_val']


argparser = argparse.ArgumentParser(
    description='Evaluation of Tracking Module')

argparser.add_argument(
    '-w',
    dest='weight',
    help='name of the trained weights')

args = argparser.parse_args()

ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, "Detection_DATA")
# Path to trained weights file
# Download this file and place in the root of your 
# project (See README file for details)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Path to LSTM Model
WEIGHTS_NAME = args.weight 
LSTM_MODEL_PATH = os.path.join(ROOT_DIR, WEIGHTS_NAME)


class LSTMInferenceConfig(trackconfig.Config):
	NAME = "OTB"
	BATCH_SIZE = 1
	TIME_STEPS = 4


############################################################
#  LSTM setting
############################################################
trackconfig = LSTMInferenceConfig()
trackconfig.display()

print("# Evaluation Weights: %s --------------- #\n" % WEIGHTS_NAME)

lstmModel = tracklib.LSTM(mode="inference", config=trackconfig, model_dir=MODEL_DIR, data_dir=DATA_DIR)
lstmModel.keras_lstm_model.load_weights(LSTM_MODEL_PATH)

lstmModel.evaluation(val_dataset=val_names, config=trackconfig, w_name=WEIGHTS_NAME)

print("# Evaluation Weights: %s --------------- #\n" % WEIGHTS_NAME)

