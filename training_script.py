from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import datetime  # for TensorBoard usage

import os
import numpy as np
import matplotlib.pyplot as plt

# SOL to activate swish activation
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation

IMAGE_SIZE = 224
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)


def swish(x):
    return K.sigmoid(x) * x


def get_callback():
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
    return [tensorboard_callback, earlystopping_callback]


def data_preparation(dataset_dir, batch_size=32):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                              validation_split=0.2)

    train_generator = datagen.flow_from_directory(dataset_dir,
                                                  target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                  batch_size=batch_size,
                                                  subset='training')

    val_generator = datagen.flow_from_directory(dataset_dir,
                                                target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                batch_size=batch_size,
                                                subset='validation')

    labels = '\n'.join(sorted(train_generator.class_indices.keys()))

    with open('labels.txt', 'w') as f:
        f.write(labels)

    return train_generator, val_generator


def start_training(train_generator, val_generator, epochs=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16,
                               kernel_size=(3, 3),  # (3, 3) means 3x3 kernal_size
                               padding='same',
                               activation=swish,
                               input_shape=IMG_SHAPE),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(32,
                               kernel_size=(3, 3),  # (3, 3) means 3x3 kernel_size
                               padding='same',
                               activation=swish),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64,
                               kernel_size=(3, 3),  # (3, 3) means 3x3 kernel size
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

    model.summary()

    print('Number of trainable variables = {}'.format(len(model.trainable_variables)))

    model.fit(train_generator,
              epochs=epochs,
              validation_data=val_generator,
              callbacks=get_callback())

    return model


def save_model(model, model_dir, model_name=None, convert_to_tflite=False):
    tf.saved_model.save(model, model_dir)

    if convert_to_tflite:
        converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)
        tflite_model = converter.convert()

        with open(model_name + '.tflite', 'wb') as f:
            f.write(tflite_model)


def run():
    get_custom_objects().update({'swish': Activation(swish)})
    train_generator, val_generator = data_preparation(dataset_dir=os.path.join('datasets', 'wrist_band_photos'))
    model = start_training(train_generator=train_generator,
                           val_generator=val_generator,
                           epochs=100)
    save_model(model=model,
               model_dir='save/fine_tuning',
               model_name='tflite_model_3',
               convert_to_tflite=True)


if __name__ == "__main__":
    run()
