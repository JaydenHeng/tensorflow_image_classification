from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import datetime #for tensorboard usage

import os
import numpy as np
import matplotlib.pyplot as plt

# from test_tflite import evaluate_tflite

# SOL to activate swish activation
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation

IMAGE_SIZE = 224
BATCH_SIZE = 32
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
train_generator = None
val_generator = None
model = None

def swish(x):
    return K.sigmoid(x) * x


def data_preparation():
    global train_generator
    global val_generator

    base_dir = os.path.join('datasets', 'wrist_band_photos')

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(base_dir,
                                                  target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                  batch_size=BATCH_SIZE,
                                                  subset='training')

    val_generator = datagen.flow_from_directory(base_dir,
                                                target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                batch_size=BATCH_SIZE,
                                                subset='validation')

    for image_batch, label_batch in train_generator:
        break
    image_batch.shape, label_batch.shape

    print('class_indices: ', train_generator.class_indices)

    labels = '\n'.join(sorted(train_generator.class_indices.keys()))

    print('labels: \n')
    print(labels)

    with open('conti_labels.txt', 'w') as f:
        f.write(labels)


def start_training():
    global model

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16,
                               kernel_size=(3, 3),  # (3, 3) means 3x3 kernal_size
                               padding='same',
                               activation=swish,
                               input_shape=IMG_SHAPE),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(32,
                               kernel_size=(3, 3),  # (3, 3) means 3x3 kernal_size
                               padding='same',
                               activation=swish),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64,
                               kernel_size=(3, 3),  # (3, 3) means 3x3 kernal_size
                               padding='same',
                               activation=swish),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=swish),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.keras.losses.BinaryCrossentropy(from_logits=True, name='binary_crossentropy'),
                           'accuracy'])

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1)

    ##################################################################
    ######################### Early Stopping #########################
    ##################################################################
    # Parameter: monitor = keep track of the quantity that is used to decide if the training should be terminated. Note: can monitor 'val_accuracy' as well
    #            min_delta = the threshold that triggers the termination. In this case, we require the accuracy should at least improve 0.00001
    #            patience = number of "no improvement epochs" to wait until training is stopped
    ##################################################################
    earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy',
                                                              min_delta=0.00001,
                                                              patience=10)

    model.summary()

    print('Number of trainable variables = {}'.format(len(model.trainable_variables)))

    epochs = 100

    model.fit(train_generator,
              epochs=epochs,
              validation_data=val_generator,
              callbacks=[tensorboard_callback, earlystopping_callback])

    # model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-5), metrics=['accuracy'])

    # model.summary()

    # print('Number of trainable variables = {}'.format(len(model.trainable_variables)))
    # print(train_generator)
    # print(val_generator)


def save_model():
    saved_model_dir = 'save/conti_fine_tuning'
    tf.saved_model.save(model, saved_model_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    tflite_model_name = 'conti_model_1'

    with open(tflite_model_name + '.tflite', 'wb') as f:
        f.write(tflite_model)


def run():
    get_custom_objects().update({'swish': Activation(swish)})
    data_preparation()
    start_training()
    save_model()
    # evaluate_tflite(tflite_model_name) #Function in test_tflite.py


if __name__ == "__main__":
    run()
