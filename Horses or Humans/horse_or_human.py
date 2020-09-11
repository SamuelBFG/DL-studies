import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import wget

# # # Download horses or humans dataset
# url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip'
# filename = wget.download(url)

# # # Extract the files
# local_zip = 'horse-or-human.zip'
# zip_ref = zipfile.ZipFile(local_zip,'r')
# zip_extractall('horse-or-human.zip')
# zip_ref.close()

actual_dir = os.path.abspath(os.getcwd())
# Directory with our training horse pictures
train_horse_dir = os.path.join(os.path.abspath(os.getcwd())+'/horses/')

# Directory with our training human pictures
train_human_dir = os.path.join(os.path.abspath(os.getcwd())+'/humans/')


train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])











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