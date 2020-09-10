import tensorflow as tf
print(tf.__version__)
import numpy as np
import matplotlib.pyplot as plt
import os
# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train_mnist():

    # Callback to stop training (Early Stopping??)
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy')>0.998): #logs.get('acc') for older versions!
                print("\nReached 99.8% accuracy so cancelling training!")
                self.model.stop_training = True
                
    callbacks = myCallback()

    # Dataset
    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0

    print('%'*30 + ' DATASET:' + '%'*30)
    print('X_train.shape:', x_train.shape)
    print('Y_train.shape:', y_train.shape)
    print('X_test.shape:', x_test.shape)
    print('Y_test.shape', y_test.shape)

    # Transforming the inputs to 4D tensor:
    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)

    # Model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    print(model.summary())

    # Fit
    history = model.fit(x_train, y_train, epochs=15, callbacks=[callbacks])


    # Return
    return history.epoch, history.history['accuracy'][-1]

if __name__ == '__main__':
    train_mnist()