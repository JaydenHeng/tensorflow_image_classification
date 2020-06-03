import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import datetime #for tensorboard usage
import os

# SOL to activate swish activation
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation


def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Activation(swish)})
#EOL

def get_callbacks(name):
  return [
  tfdocs.modelling.EpochDots(),
  tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentrophy', patience=200),
  tf.keras.get_callbacks.TensorBoard(logdir/name),
  ]

base_dir = os.path.join('datasets', 'wrist_band_photos')

IMAGE_SIZE = 224
BATCH_SIZE = 32

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(base_dir, target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE, subset='training')

val_generator = datagen.flow_from_directory(base_dir, target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE, subset='validation')

HP_CONVO_FILTER = hp.HParam('filter', hp.Discrete([64, 128]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_CONVO_FILTER, HP_DROPOUT],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
    )


def train_test_model(hparams):

    IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, alpha=1.0, include_top=False,
                                                   weights='imagenet')

    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        # tf.keras.layers.Conv2D(32, 3, activation='relu'),
        # tf.keras.layers.Conv2D(64, 3, activation='relu'),
        # tf.keras.layers.Conv2D(128, 3, activation='relu'),
        # tf.keras.layers.Conv2D(128, 3, activation=swish), #Work Best
        tf.keras.layers.Conv2D(hparams[HP_CONVO_FILTER], kernel_size=(5, 5), padding='same', activation=swish),
        # (5,5) means 5x5 kernal_size
        tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
        # tf.keras.layers.GlobalMaxPooling2D(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-5), metrics=['accuracy'])

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    model.fit(train_generator, epochs=3, callbacks=[tf.keras.callbacks.TensorBoard(log_dir),  # log metrics
        hp.KerasCallback(log_dir, hparams),  # log hparams
    ])  # Run with 1 epoch to speed things up for demo purposes
    _, accuracy = model.evaluate(val_generator)
    return accuracy


def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

session_num = 0

# for num_units in HP_NUM_UNITS.domain.values:
#     for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
#         for optimizer in HP_OPTIMIZER.domain.values:
#             hparams = {
#                 HP_NUM_UNITS: num_units,
#                 HP_DROPOUT: dropout_rate,
#                 HP_OPTIMIZER: optimizer,
#             }
#             run_name = "run-%d" % session_num
#             print('--- Starting trial: %s' % run_name)
#             print({h.name: hparams[h] for h in hparams})
#             run('logs/hparam_tuning/' + run_name, hparams)
#             session_num += 1

for convo_filter in HP_CONVO_FILTER.domain.values:
    for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
        hparams = {
            HP_CONVO_FILTER: convo_filter,
            HP_DROPOUT: dropout_rate,
        }
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        run('logs/hparam_tuning/' + run_name, hparams)
        session_num += 1
