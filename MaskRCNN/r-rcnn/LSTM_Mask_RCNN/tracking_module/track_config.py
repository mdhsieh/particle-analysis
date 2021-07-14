"""
Recurrent Region CNN
Base Configurations class.

Copyright (c) 2018 Chen-En Chung
Licensed under the MIT License (see LICENSE for details)
Written by Chen-En, Chung
"""

import math
import numpy as np

# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):
    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running
	NAME = None

	# Number of GPUs to use. 
	GPU_COUNT = 1

	# Effective batch size
	BATCH_SIZE = 32
	
	# LSTM layer 1 cell size
	L1_CELL_SIZE = 4096

	# Tracking length (frame)
	TIME_STEPS = 4

	# MRCNN feature map size
	FEATURE_SIZE = 1024

	# MRCNN detected bbox size
	MRCNNBBOX_SIZE = 4

	# LSTM input size
	INPUT_SIZE = FEATURE_SIZE + MRCNNBBOX_SIZE

	# LSTM output size
	OUTPUT_SIZE = 4 # Bbox 4 coordinate

	# Learning rate and momentum
	LEARNING_RATE = 1e-5

	# Number of training iteration per epoch
	STEPS_PER_EPOCH = 1000

	# Number of epoch
	EPOCHS = 50

	# Number of validation steps to run at the end of every training epoch.
	# A bigger number improves accuracy of validation stats, but slows
	# down the training.
	VALIDATION_STEPS = 50

	# Threshold for filtering MRCNN results
	THRESHOLD = 0.15

	def display(self):
		"""Display Configuration values."""
		print("\nLSTM Configurations:")
		for a in dir(self):
			if not a.startswith("__") and not callable(getattr(self, a)):
				print("{:30} {}".format(a, getattr(self, a)))
		print("\n")