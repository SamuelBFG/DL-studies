import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
from tf.keras.preprocessing.image import ImageDataGenerator
local_zip = '/tmp/horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip,'r')
zip_extractall('/tmp/horse-or-human')
train_datagen = ImageDataGenerator(rescale=1./255) # Normalize the data
 
# Important: Point to directory that contains the sub-directories
train_generator = train_datagen.flow_from_directory(
	train_dir,
	target_size=(300,300),
	batch_size=128,
	class_mode='binary')

validation_generator = train_datagen.flow_from_directory(
	validation_dir,
	target_size=(300,300),
	batch_size=128,
	class_mode='binary')

# Important: diferent .fit method cuz we're using GENERATORS.
history = model.fit_generator(
	train_generator, #streams the images from the training directory
	steps_per_epoch=8, #1024 images 128 at time, 1024/128=8 batches
	epochs=15, 
	validation_data=validation_generator,
	validation_steps=8, #256 images 32 at time, 256/32=8 batches
	verbose=2)