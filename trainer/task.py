'''
Train PredNet on KITTI sequences. (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''

import os
import numpy as np
np.random.seed(123)
from six.moves import cPickle
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint,EarlyStopping
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from prednet import PredNet
from data_utils import SequenceGenerator
from kitti_settings import *
import scipy.misc
from keras.models import load_model
from tensorflow.python.lib.io import file_io
from google.cloud import storage

def main():
	client = storage.Client()
	bucket = client.bucket('meniscus_cloud_data_1000')

	val_recordings = [('clouds', 'validation')]
	test_recordings = [('clouds', 'testing')]
	categories = 'clouds'

	desired_im_sz = (1000, 1000)

	#### Trainiing data #######

	training_dataset =[]
	for blob in bucket.list_blobs(prefix='kitti_data/clouds/training'):
	  training_dataset.append(blob.name)

	training_images = []
	for i in len(training_dataset):
		name = 'kitti_data/clouds/validation/'+ training_dataset[i]
		for blob in bucket.list_blobs(prefix='kitti_data/clouds/training/'+training_dataset[i]):
			#training_images += name + '/' + blob.name
			training_images += blob.id
			
	X = np.zeros((len(training_images),) + desired_im_sz + (3,), np.uint8)

	for i, filename in enumerate(training_images):
		with open(filename, "wb") as file_obj:
			im_file = blob.download_to_file(file_obj)
			im = scipy.misc.imread(im_file)
			X[i] = im	
	train_sources = training_images
	train_file = X



	#### Validation data #######
	validation_dataset =[]
	for blob in bucket.list_blobs(prefix='kitti_data/clouds/validation'):
	  validation_dataset.append(blob.name)

	validation_images = []
	for i in len(validation_dataset):
		name = 'kitti_data/clouds/validation/'+ validation_dataset[i]
		for blob in bucket.list_blobs(prefix= name):
			#validation_images_blob += name + '/' + blob.name
			validation_images += blob.id
			
	X = np.zeros((len(validation_images),) + desired_im_sz + (3,), np.uint8)	

	for i, im_file in enumerate(validation_images):
		with open(filename, "wb") as file_obj:
			im_file = blob.download_to_file(file_obj)
			im = scipy.misc.imread(im_file)
			X[i] = im	
		
	val_sources  = validation_images
	val_file = X
			

		
			

	save_model = True  # if weights will be saved
	weights_file = WEIGHTS_DIR + 'prednet_kitti_weights.hdf5' # where weights will be saved
	json_file = WEIGHTS_DIR + 'prednet_kitti_model.json'
	saved_models = WEIGHTS_DIR + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'




	#Training parameters
	nb_epoch = 50
	samples_per_epoch = 28

	batch_size = 8
	N_seq_val = 4 # number of sequences to use for validation
	# number of timesteps used for sequences in training
	nt = 4 

	lr = 0.001 #learning rate
	up_lr = 0.0001 #learinig rate is updated to this new value
	up_lr_ep = 40 #point at which learinig rate should be updated



	# Model parameters
	#n_channels, im_height, im_width = (3, 128, 160)
	n_channels, im_height, im_width = (3, 1000, 1000)
	input_shape = (n_channels, im_height, im_width) if K.image_data_format() == 'channels_first' else (im_height, im_width, n_channels)
	stack_sizes = (n_channels, 48, 96, 192)
	#Lstm stack_sizes
	R_stack_sizes = stack_sizes
	#convolutional filter size
	A_filt_sizes = (3, 3, 3)
	#prdiction convloutinal filter size
	Ahat_filt_sizes = (3, 3, 3, 3)
	# recurrent convolution filter size
	R_filt_sizes = (3, 3, 3, 3)
	# weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
	layer_loss_weights = np.array([1., 0., 0., 0.])  
	layer_loss_weights = np.expand_dims(layer_loss_weights, 1)


	# equally weight all timesteps except the first
	time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))  
	time_loss_weights[0] = 0

	#-----------------------------------------Arcitecture---------------------------------------------------#
	prednet = PredNet(stack_sizes, R_stack_sizes,
					  A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
					  output_mode='error', return_sequences=True)

	#-----------------------------------------Layers--------------------------------------------------------#
	#initializing a tensor for input                                                                        #
	inputs = Input(shape=(nt,) + input_shape)                                                               #
	# errors will be (batch_size, nt, nb_layers)                                                            #
	errors = prednet(inputs)                                                                                #
	# calculate weighted error by layer                                                                     #
	errors_by_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)],  #
									  trainable=False)(errors)                                              #
	# will be (batch_size, nt)                                                                              #
	errors_by_time = Flatten()(errors_by_time)                                                              #
	# weight errors by time                                                                                 #
	# dense() creates a densely connected network                                                           #                                                                              #
	final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)      #
	#-------------------------------------------------------------------------------------------------------#


	#----------------------------------Create model---------------------------------------------------------#
	model = Model(inputs=inputs, outputs=final_errors)   
	#model = load_model(weights_file,custom_objects={'prednet': PredNet})                                                   #
	model.compile(loss='mean_absolute_error', optimizer='adam')  
	#model.load_weights(weights_file)                                              
	#-------------------------------------------------------------------------------------------------------#


	#-------------------------------------Data Preprocessing------------------------------------------------#
	train_generator = SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size, shuffle=True) #
	val_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val)    #
	#-------------------------------------------------------------------------------------------------------#




	#-------------------------------------Callback functions for training--------------------------------------#
	# start with lesrning rate of 0.001 and then drop to 0.0001 after 75 epochs                                #
	lr_schedule = lambda epoch: lr if epoch < up_lr_ep else up_lr                                              #
	callbacks = [LearningRateScheduler(lr_schedule)]                                                           #
																											   #
	#save model best model check points                                                                        #
	# if save_model:                                                                                             #
		# if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)                                              #
		# callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))  
		#
	callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))

	#Tensorboard for visualization                                                                             #
	tb = TensorBoard(log_dir=GRAPH_DIR,batch_size=batch_size,histogram_freq=2,write_graph=True,write_images=True)                         
	tb.set_model(model)
	callbacks.append(tb)  

	checkPoints = ModelCheckpoint(saved_models, monitor='val_loss', 
					verbose=0, save_best_only=True, 
					save_weights_only=False, 
					mode='auto', period=1)                                                                                     #

	callbacks.append(checkPoints)

	#earlyStops = EarlyStopping(monitor='val_loss', min_delta=0, 
	#					patience=0, verbose=0, mode='auto')

	#callbacks.append(earlyStops)
	#--------------------------------------------------------------------------------------------------------------#


	#-------------------------------------Training-----------------------------------------------------------------#
	history = model.fit_generator(train_generator, samples_per_epoch / batch_size, nb_epoch, callbacks=callbacks,  #
					validation_data=val_generator, validation_steps=N_seq_val / batch_size)                        #
																												   #
	#save model				                                                                                       #
	# if save_model:                                                                                                 #
		# json_string = model.to_json()                                                                              #
		# with open(json_file, "w") as f:                                                                            #
			# f.write(json_string) 

			
	json_string = model.to_json() 
	with file_io.FileIO(json_string, mode='r') as input_f:
		with file_io.FileIO(json_file, mode='w+') as output_f:
			output_f.write(input_f.read())
			
	#------------------------------------------------------------------------------------------------------------- #


##Running the app
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    # # Input Arguments
    # parser.add_argument(
      # '--job-dir',
      # help='GCS location to write checkpoints and export models',
      # required=True
    # )
    # args = parser.parse_args()
    # arguments = args.__dict__

    main() 