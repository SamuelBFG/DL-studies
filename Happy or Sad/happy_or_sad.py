import os
import zipfile
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import path, getcwd, chdir
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


############################################
################ DIR PATHS #################
############################################
# path = f"{getcwd()}/../tmp2/happy-or-sad.zip"
path = r'happy-or-sad.zip'
zip_ref = zipfile.ZipFile(path, 'r')
zip_ref.extractall("/tmp/h-or-s")
zip_ref.close()

# Directory with our training horse pictures
train_happy_dir = os.path.join('/tmp/h-or-s/happy')
train_happy_names = os.listdir(train_happy_dir)
# Directory with our training human pictures
train_sad_dir = os.path.join('/tmp/h-or-s/sad')
train_sad_names = os.listdir(train_sad_dir)


############################################
############ PLOTTING SAMPLES ##############
############################################
# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_happy_pix = [os.path.join(train_happy_dir, fname) 
                for fname in train_happy_names[pic_index-8:pic_index]]
next_sad_pix = [os.path.join(train_sad_dir, fname) 
                for fname in train_sad_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_happy_pix+next_sad_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

############################################
################## MODEL ###################
############################################

def train_happy_sad_model():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):
         def on_epoch_end(self, epoch, logs={}):
                if(logs.get('accuracy')>DESIRED_ACCURACY): # logs.get('acc') for older versions
                    print('\nReached 99.9% accuracy so cancelling the training')
                    self.model.stop_training = True

    callbacks = myCallback()
    
    # This Code Block should Define and Compile the Model. Please assume the images are 150 X 150 in your implementation.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150,150,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # PRINT SUMMARY
    print(model.summary())
    
    

    model.compile(loss='binary_crossentropy',
                 optimizer=RMSprop(lr=0.001),
                  metrics=['accuracy'])
       

    
    ################## DataGenerator ###################
    
    train_datagen = ImageDataGenerator(rescale=1./255)

    # target_size of 150 X 150.
    train_generator = train_datagen.flow_from_directory(
        '/tmp/h-or-s/',
        target_size=(150,150),
        batch_size=16,
        class_mode='binary')
    # Expected output: 'Found 80 images belonging to 2 classes'

    # model fitting
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=5,
        epochs=20,
        verbose=2,
    callbacks=[callbacks])

    return history.history['accuracy'][-1]

if __name__ == '__main__':
    train_happy_sad_model()