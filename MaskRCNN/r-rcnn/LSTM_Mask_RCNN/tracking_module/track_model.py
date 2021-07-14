"""
Recurrent Region CNN
The main tracking module implemenetation.

Copyright (c) 2018 Chen-En Chung
Licensed under the MIT License (see LICENSE for details)
Written by Chen-En, Chung
"""

import numpy as np
import random
import os
import scipy.misc
import skimage.io
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM
import keras.optimizers as KO
import utils.utils as utils
import time
from matplotlib import pyplot as plt
from keras.utils.vis_utils import plot_model
import json

np.random.seed(9527)  # for reproducibility

def get_rand_name(namelist, b_size):
	result = []
	while b_size-len(result) > len(namelist):
		random.shuffle(namelist)
		result.extend(namelist)
	if b_size > len(result):
		result.extend(random.sample(namelist, b_size-len(result)))

	return result


def data_generator(dataset, config, data_dir):
	X_coord = np.zeros((config.BATCH_SIZE, config.TIME_STEPS, config.MRCNNBBOX_SIZE))
	X_feat = np.zeros((config.BATCH_SIZE, config.TIME_STEPS, config.FEATURE_SIZE))
	Y_coord = np.zeros((config.BATCH_SIZE, config.TIME_STEPS, config.OUTPUT_SIZE))

	# Keras requires a generator to run indefinately.
	while True:
		name_batch = get_rand_name(dataset, config.BATCH_SIZE)
		# From each video, random choose TIME_STEPS of consecutive frames
		for b, name in enumerate(name_batch):
			# Read bbox & annotation
			bbox_name = os.path.join(data_dir, "bbox", name+".npy")
			x_bbox = np.load(bbox_name)
			feat_name = os.path.join(data_dir, "feature", name+".npy")
			x_feat = np.load(feat_name)
			y_name = os.path.join(data_dir, "bbox_GT", name+".npy")
			y = np.load(y_name)

			# Random choose start frame
			start = random.randint(0, len(y)-config.TIME_STEPS)
			
			# Keep in input array
			X_coord[b] = x_bbox[start:start+config.TIME_STEPS]
			X_feat[b] = x_feat[start:start+config.TIME_STEPS]
			Y_coord[b] = y[start:start+config.TIME_STEPS]

		yield [X_coord, X_feat], Y_coord


def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typicallly: [N, 4], but could be any shape.
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)

    return K.mean(loss, axis=-1) * 100


############################################################
#  LSTM Class 
############################################################
class LSTM():
	def __init__(self, mode, config, model_dir, data_dir=None, dataset_dir=None, w_name=None):
		assert mode in ['training', 'inference']
		self.mode = mode
		self.config = config
		self.model_dir= model_dir
		self.data_dir = data_dir
		self.dataset_dir = dataset_dir
		self.set_log_dir(w_name)
		self.keras_lstm_model = self.build(mode=mode, config=config)


	def build(self, mode, config):
		print("# Building LSTM %s model --------------- #\n" % (mode))

		# Model graph		
		# Bbox coord: [Batch, TS, 4]
		input_coord = KL.Input(batch_shape=(config.BATCH_SIZE, config.TIME_STEPS, config.MRCNNBBOX_SIZE))
		# Feature map: [Batch, TS, 1024]
		input_feat = KL.Input(batch_shape=(config.BATCH_SIZE, config.TIME_STEPS, config.FEATURE_SIZE))

		# LSTM
		if mode == 'training':
			# Separate
			x_coord = KL.CuDNNLSTM(
					units=config.L1_CELL_SIZE,
					return_sequences=True,
					stateful=False
					)(input_coord)
			x_feat = KL.CuDNNLSTM(
					units=config.L1_CELL_SIZE,
					return_sequences=True,
					stateful=False
					)(input_feat)
			x_coord = KL.TimeDistributed(KL.Dense(2048, activation='relu'))(x_coord)
			x_coord = KL.TimeDistributed(KL.Dense(1024, activation='relu'))(x_coord)
			x_feat = KL.TimeDistributed(KL.Dense(2048, activation='relu'))(x_feat)
			x_feat = KL.TimeDistributed(KL.Dense(1024, activation='relu'))(x_feat)
			x = KL.Concatenate()([x_coord, x_feat])
			delta = KL.TimeDistributed(KL.Dense(config.OUTPUT_SIZE, activation='tanh'))(x)
			out_bbox = KL.Add()([input_coord, delta])

		elif mode == 'inference':
			# Separate
			x_coord = KL.CuDNNLSTM(
					units=config.L1_CELL_SIZE,
					return_sequences=False,
					stateful=False
					)(input_coord)
			x_feat = KL.CuDNNLSTM(
					units=config.L1_CELL_SIZE,
					return_sequences=False,
					stateful=False
					)(input_feat)
			x_coord = KL.Dense(2048, activation='relu')(x_coord)
			x_coord = KL.Dense(1024, activation='relu')(x_coord)	
			x_feat = KL.Dense(2048, activation='relu')(x_feat)
			x_feat = KL.Dense(1024, activation='relu')(x_feat)
			x = KL.Concatenate()([x_coord, x_feat])
			delta = KL.Dense(config.OUTPUT_SIZE, activation='tanh')(x)
			last_coord = KL.Lambda(lambda x: x[:, config.TIME_STEPS-1, :], output_shape=(4, ))(input_coord)
			out_bbox = KL.Add()([last_coord, delta])

		# Create model
		model = KM.Model(inputs=[input_coord, input_feat],
						 outputs=out_bbox,
						 name='RRCNN')
		model.summary()
		if mode == 'training':
			# Initial loss function and optimizer
			adam = KO.Adam(config.LEARNING_RATE)
			model.compile(optimizer=adam, loss=smooth_l1_loss)
			plot_model(model, to_file='RRCNN.png', show_shapes=True, show_layer_names=False)

		return model


	def set_log_dir(self, weights_name):
		self.epoch = 0
		now = time.strftime("%Y%m%d-%H", time.localtime())
		# Directory for training logs
		self.log_dir = os.path.join(self.model_dir, "{}_{}".format(
			self.config.NAME, now))

		# Path to save after each epoch. Include placeholders that get filled by Keras.
		if weights_name:
			self.checkpoint_path = os.path.join(self.log_dir, "{}_*epoch*.h5".format(
				weights_name))
			self.checkpoint_path = self.checkpoint_path.replace(
				"*epoch*", "{epoch:04d}")
		else:
			self.checkpoint_path = os.path.join(self.log_dir, "lstm_*epoch*.h5")
			self.checkpoint_path = self.checkpoint_path.replace(
				"*epoch*", "{epoch:04d}")


	def train(self, train_dataset, val_dataset):
		print("# Training --------------- #\n")

		# Data generators
		train_generator = data_generator(train_dataset, self.config, self.data_dir)
		val_generator = data_generator(val_dataset, self.config, self.data_dir)

		# Callbacks
		callbacks = [
			keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
			keras.callbacks.ModelCheckpoint(self.checkpoint_path,
										verbose=0, save_weights_only=True)
		]

		# Train
		model_history = self.keras_lstm_model.fit_generator(
			generator=train_generator,
			initial_epoch=self.epoch,
			epochs=self.config.EPOCHS,
			steps_per_epoch=self.config.STEPS_PER_EPOCH,
			callbacks=callbacks,
			validation_data=next(val_generator),
			validation_steps=self.config.VALIDATION_STEPS, 
			use_multiprocessing=True
		)
		self.epoch = max(self.epoch, self.config.EPOCHS)


	def detect(self, lstmInput):
		result = self.keras_lstm_model.predict(lstmInput, batch_size=self.config.BATCH_SIZE)

		return result


	def evaluation(self, val_dataset, config, w_name):
		result_dir = os.path.join(os.getcwd(), "RRCNN_results", w_name[:-3], "bbox")
		if not os.path.exists(result_dir):
			os.makedirs(result_dir)

		print("\n# Valildation Evaluation --------------- #\n")
		total_iou = 0
		video_count = len(val_dataset)
		aos = 0
		img_count = 0

		for idx, name in enumerate(val_dataset):
			bbox_name = os.path.join(self.data_dir, "bbox", name+".npy")
			bbox = np.load(bbox_name)
			feat_name = os.path.join(self.data_dir, "Feature", name+".npy")
			feat = np.load(feat_name)
			y_name = os.path.join(self.data_dir, "bbox_GT", name+".npy")
			y = np.load(y_name)

			result_name = os.path.join(result_dir, name)
			result = np.zeros((bbox.shape[0], 4))
			folder_iou = 0
			print("# Processing folder: %s, %d images. #" % (name, bbox.shape[0]))
			# Using LSTM to tracking one video
			for img_id in range(bbox.shape[0]+1-config.TIME_STEPS):
				# Input Bbox
				X_coord = np.zeros((1, config.TIME_STEPS, config.MRCNNBBOX_SIZE))
				if img_id == 0:
					X_coord[0] = bbox[img_id:img_id+config.TIME_STEPS]
				elif config.TIME_STEPS == 2:
					X_coord[0, 0] = last_result
					X_coord[0, 1] = bbox[img_id+1]
				else:
					X_coord[0, :config.TIME_STEPS-2] = last_input[1:config.TIME_STEPS-1]
					X_coord[0, config.TIME_STEPS-2] = last_result
					X_coord[0, config.TIME_STEPS-1] = bbox[img_id+config.TIME_STEPS-1]
				
				# Input Feature
				X_feat = feat[img_id:img_id+config.TIME_STEPS].reshape(1, config.TIME_STEPS, config.FEATURE_SIZE)

				# Run tracking
				lstm_bbox = self.detect(lstmInput=[X_coord, X_feat])
				lstm_bbox = lstm_bbox[0]

				last_result = lstm_bbox
				last_input = X_coord[0]

				# Keep result of LSTM
				result[img_id+config.TIME_STEPS-1, :] = lstm_bbox
				# Calculate iou of result, exclude 1st image
				lstm_bbox = utils.bbox_toyxyx(lstm_bbox)
				anno_bbox = utils.bbox_toyxyx(y[img_id+config.TIME_STEPS-1, :])
				iou = utils.eval_bbox_iou(anno_bbox, lstm_bbox)
				folder_iou += iou

			# Store LSTM results for each video
			np.save(result_name, result)
			# Compute iou for each video
			folder_iou /= (bbox.shape[0]-config.TIME_STEPS+1)
			img_count += (bbox.shape[0]-config.TIME_STEPS+1)
			total_iou += folder_iou
			print("%s aos: %.3f\n" % (name, folder_iou))

		# Compute Average Overlap Score
		aos = total_iou / video_count

		print("# Valildation Final AOS: %.3f --------------- #\n" % (aos))

		return aos
