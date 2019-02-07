'''
Fine-tune PredNet model trained for t+1 prediction for up to t+5 prediction.
'''

import os
import numpy as np
np.random.seed(123)

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input
from keras.callbacks import LearningRateScheduler, ModelCheckpoint,EarlyStopping
from keras.callbacks import TensorBoard
from prednet import PredNet
from data_utils import SequenceGenerator
from kitti_settings import *
import scipy.misc
from keras.models import load_model

# Define loss as MAE of frame predictions after t=0
# It doesn't make sense to compute loss on error representation, since the error isn't wrt ground truth when extrapolating.
def extrap_loss(y_true, y_hat):
    y_true = y_true[:, 1:]
    y_hat = y_hat[:, 1:]
    return 0.5 * K.mean(K.abs(y_true - y_hat), axis=-1)  # 0.5 to match scale of loss when trained in error mode (positive and negative errors split)

#nt = 15
#extrap_start_time = 10  # starting at this time step, the prediction from the previous time step will be treated as the actual input
#orig_weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')  # original t+1 weights
#orig_json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')

nt = 10
extrap_start_time = 6  # starting at this time step, the prediction from the previous time step will be treated as the actual input
orig_weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')  # original t+1 weights
orig_json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')

save_model = True
extrap_weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights-extrapfinetuned.hdf5')  # where new weights will be saved
extrap_json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model-extrapfinetuned.json')

val_recordings = [('clouds', 'Validation')]
test_recordings = [('clouds', 'Testing')]
categories = 'clouds'


#desired_im_sz = (128, 160)
desired_im_sz = (280, 400)


splits = {s: [] for s in ['train', 'test', 'val']}
splits['val'] = val_recordings
splits['test'] = test_recordings
not_train = splits['val'] + splits['test']
#for c in categories: 
# Randomly assign recordings to training and testing. Cross-validation done across entire recordings.
c = categories
c_dir = os.path.join(DATA_DIR, 'raw', c + '/')

_, folders, _ = next(os.walk(c_dir))
splits['train'] += [(c, f) for f in folders if (c, f) not in not_train]
	
for split in splits:
	im_list = []
	source_list = []  
	for category, folder in splits[split]:
		im_dir = os.path.join(DATA_DIR, 'raw/', category, folder)
		print(im_dir)
		for root,dirs,files in os.walk(im_dir):  
					
			im_list += [root + '\\' + f for f in sorted(files)]
			source_list += [category + '-' + folder] * len(files)
		
	
	filename = split+'set.txt'
	mfile = open(filename, 'w')
	for im in im_list:
		mfile.write("%s\n" % im)	
	print('Creating' + split + ' data: ' + str(len(im_list)) + ' images')
	X = np.zeros((len(im_list),) + desired_im_sz + (3,), np.uint8)
	

	for i, im_file in enumerate(im_list):
		im = scipy.misc.imread(im_file)
		X[i] = im
		
	if(split == 'train'):
		train_sources = source_list
		train_file = X
		
		
	if(split == 'val'):
		val_sources  = source_list
		val_file = X


# Data files
#train_file = os.path.join(DATA_DIR, 'X_train.hkl')
#train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
#val_file = os.path.join(DATA_DIR, 'X_val.hkl')
#val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')

#Training parameters


nb_epoch = 50
samples_per_epoch = 28

batch_size = 4
N_seq_val = 4 # number of sequences to use for validation
# number of timesteps used for sequences in training
#nt = 4 

lr = 0.001 #learning rate
up_lr = 0.0001 #learinig rate is updated to this new value
up_lr_ep = 40

# Training parameters
#nb_epoch = 150
#batch_size = 4
#samples_per_epoch = 500
#N_seq_val = 100  # number of sequences to use for validation

# Load t+1 model
f = open(orig_json_file, 'r')
json_string = f.read()
f.close()
orig_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
orig_model.load_weights(orig_weights_file)

layer_config = orig_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
layer_config['extrap_start_time'] = extrap_start_time
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
prednet = PredNet(weights=orig_model.layers[1].get_weights(), **layer_config)

input_shape = list(orig_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt

inputs = Input(input_shape)
predictions = prednet(inputs)
model = Model(inputs=inputs, outputs=predictions)
model.compile(loss=extrap_loss, optimizer='adam')

train_generator = SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size, shuffle=True, output_mode='prediction')
val_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val, output_mode='prediction')

#lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
lr_schedule = lambda epoch: lr if epoch < up_lr_ep else up_lr                                              #
callbacks = [LearningRateScheduler(lr_schedule)]     


if save_model:
    if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)
    callbacks.append(ModelCheckpoint(filepath=extrap_weights_file, monitor='val_loss', save_best_only=True))
	
#tb = TensorBoard(log_dir="./Graph",batch_size=batch_size,histogram_freq=2,write_graph=True)                         
#tb.set_model(model)
#callbacks.append(tb)

#earlyStops = EarlyStopping(monitor='val_loss', min_delta=0, 
#					patience=0, verbose=0, mode='auto')

#callbacks.append(earlyStops)

if save_model:
    json_string = model.to_json()
    with open(extrap_json_file, "w") as f:
        f.write(json_string)

history = model.fit_generator(train_generator, samples_per_epoch / batch_size, nb_epoch, callbacks=callbacks,
                validation_data=val_generator, validation_steps=N_seq_val / batch_size)


