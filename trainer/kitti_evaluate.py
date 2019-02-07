'''
Evaluate trained PredNet on KITTI sequences.
Calculates mean-squared error and plots predictions.
'''

import os
import numpy as np

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from prednet import PredNet
from data_utils import SequenceGenerator
from kitti_settings import *
import scipy.misc

from apscheduler.schedulers.blocking import BlockingScheduler
import time
import datetime

import shutil
import glob

import logging
#logging.basicConfig(filename='E:/Workspace/GeneratePrediction/predictor/prediction.log',format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.DEBUG)

def sorted_dir(folder):
    def getmtime(name):
        path = os.path.join(folder, name)
        return os.path.getmtime(path)

    return sorted(os.listdir(folder), key=getmtime)
	
def createPrediction():
	
	logging.info('Started making prediction')
	ts = time.time()
	st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
	locationOfImages = "E:/Workspace/GeneratePrediction/predictor/data"
	locationOfPredictedImages = "E:/Workspace/GeneratePrediction/predictor/result"
	outputLocation = "E:/Source/Output"
	inputLocation = "E:/Source/Input"
	
	# Number of predictions to be made
	noOfPredictions = 6
		
	filename = None
	
    # Move new files to the correct location	
	for file in os.listdir(inputLocation):
		if file.endswith('.png'): 
			shutil.copy2(os.path.join(inputLocation, file), locationOfImages)
			filename = os.path.basename(file)
			logging.info("Moving file " + filename + " to " + locationOfImages)
			os.remove(os.path.join(inputLocation, file))
	
	# Number of images needed to make prediction		
	if(filename):
		target = 4
	else:
		target = 99
	
	# Remove any files from the previous day as these cannot be used in today's prediction
	currentweekday = datetime.datetime.fromtimestamp(ts).weekday()
	for file in os.listdir(locationOfImages):
		if datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(locationOfImages, file))).weekday() != currentweekday:
			logging.info("Removing file " + file + " from a previous day")
			os.remove(os.path.join(locationOfImages, file))	
				
	numberOfFilesInFolder = len(glob.glob(os.path.join(locationOfImages, "*.png")))		
			
	# If the number of images in the source folder reach the target, move them to testing folder and start making predictions
	if numberOfFilesInFolder >= target:		
	
		print("There are " + str(numberOfFilesInFolder) + " files in '" + locationOfImages + "'")		
		
		# Loop through based on the number of predictions to be made		
		for i in range(noOfPredictions):
			
			# Number of files in the folder that contains the normal source images
			numberOfFilesinTestingFolder = len(os.listdir(locationOfImages))
			
			# Run prediction
			evaluate(numberOfFilesinTestingFolder)
			
			# Number of images in the predicted folder
			numberOfPredictedImages = len(os.listdir(locationOfPredictedImages))
			
			# Copy the most recent prediction into the source folder
			list_of_files = glob.glob(os.path.join(locationOfPredictedImages, "*.*"))
			lastImage = max(list_of_files, key=os.path.getmtime)
			logging.info("The latest predicted file is '" + lastImage + "' and will be moved to the source location.")
			shutil.copy2(lastImage, locationOfImages)
			
		# Remove all files in the prediction folder	as these are no longer required - all the files that
		# we require are in the source directory
		for file in os.listdir(locationOfPredictedImages):
			os.remove(os.path.join(locationOfPredictedImages, file))
		
		# Move the required images to the output location
		c = 1
		list_of_files = glob.glob(os.path.join(locationOfImages, "*.*"))
		logging.info('Number of files in' + locationOfImages + '= ' + str(len(list_of_files)))
		for name in sorted_dir(locationOfImages) :		
			if "img" in name:				
				imgName = os.path.splitext(filename)[0] + '_' + str(c * 15) + '.png'
				logging.info("Renaming file '" + name + "' to '" + imgName + "'.")
				os.rename(os.path.join(locationOfImages, name), os.path.join(locationOfImages, imgName))
				shutil.copy2(os.path.join(locationOfImages, imgName), outputLocation)
				os.remove(os.path.join(locationOfImages, imgName))
				c = c + 1
		
		# Remove the first image in the source folder
				
		list_of_files = glob.glob(os.path.join(locationOfImages, "*.*"))
		if(len(list_of_files) >= 6):
			firstImage = min(list_of_files, key=os.path.getmtime)
			logging.info("The first source image '" + firstImage + "' will be removed.")
			os.remove(firstImage)

	else:
		logging.info("Insufficent images to create prediction")
		pass		
		
def evaluate(numberOfImages):
	n_plot = 1
	batch_size = 4
	nt = numberOfImages

	desired_im_sz = (280, 400)

	weights_file = 'E:/Workspace/GeneratePrediction/predictor/model/prednet_kitti_weights-extrapfinetuned.hdf5'
	json_file = 'E:/Workspace/GeneratePrediction/predictor/model/prednet_kitti_model-extrapfinetuned.json'
	DATA_DIR = "E:/Workspace/GeneratePrediction/predictor/data"
	RESULTS_SAVE_DIR = "E:/Workspace/GeneratePrediction/predictor/result/"
		
	im_list = []
	source_list = [] 
		
	for root,dirs,files in os.walk(DATA_DIR):  			
		im_list += [root + '//' + f for f in sorted(files)]
		#source_list += [category + '-' + folder] * len(files)	
		source_list += [files] * len(files)
	
	print('Creating data: ' + str(len(im_list)) + ' images')
	X = np.zeros((len(im_list),) + desired_im_sz + (3,), np.uint8)

	for i, im_file in enumerate(im_list):
		im = scipy.misc.imread(im_file)
		X[i] = im
		
	test_sources = source_list
	test_file = X
			
	# Load trained model
	f = open(json_file, 'r')
	json_string = f.read()
	f.close()
	train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
	train_model.load_weights(weights_file)

	# Create testing model (to output predictions)
	layer_config = train_model.layers[1].get_config()
	layer_config['output_mode'] = 'prediction'
	data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
	test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
	input_shape = list(train_model.layers[0].batch_input_shape[1:])
	input_shape[0] = nt
	inputs = Input(shape=tuple(input_shape))
	predictions = test_prednet(inputs)
	test_model = Model(inputs=inputs, outputs=predictions)
	
	test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique', data_format=data_format)

	X_test = test_generator.create_all()
	X_hat = test_model.predict(X_test, batch_size)
	K.clear_session()
	if data_format == 'channels_first':
		X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
		X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))
	
	# Uncomment to retrive mse score for the model and previcous image	
	# Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
	# mse_model = np.mean( (X_test[:, 1:] - X_hat[:, 1:])**2 )  # look at all timesteps except the first
	# mse_prev = np.mean( (X_test[:, :-1] - X_test[:, 1:])**2 )

	# Plot some predictions
	aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
	pred_save_dir = RESULTS_SAVE_DIR
	print (pred_save_dir)
	plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]
	for i in plot_idx:
		for t in range(nt):
			final_img =  pred_save_dir +  'img' + str(t) + '.png'
			print(final_img)
			scipy.misc.imsave(final_img, X_hat[i,t])

# Create a set of predictions	
createPrediction()



