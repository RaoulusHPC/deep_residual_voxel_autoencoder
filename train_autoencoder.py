
#from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import layers, losses
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import numpy as np
from pathlib import Path
from random import *

from efficientnets import efficientnets3d_v2

train_efficientnet = True

class TrainingArguments:
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-3
    adam_eps: float = 1e-5


def get_abc_dataset(tfrecords: list[str]):
  raw_component_dataset = tf.data.TFRecordDataset(tfrecords, compression_type='GZIP')

  # Create a dictionary describing the features.
  component_feature_description = {
    'component_raw': tf.io.FixedLenFeature([], tf.string),
  }

  def read_tfrecord(serialized_example):
    # Parse the input tf.train.Example proto using the dictionary above.
    example = tf.io.parse_single_example(serialized_example, component_feature_description)
    component = tf.io.parse_tensor(example['component_raw'], out_type=bool)
    component = tf.cast(component, 'float32')
    component = tf.expand_dims(component, axis=-1)
    return component, component

  dataset = raw_component_dataset.map(read_tfrecord)
  return dataset

class ConvBlock(tf.keras.layers.Layer):
  def __init__(self, filters, kernel_size, strides, padding, activation, skip_connection=False):
    super(ConvBlock, self).__init__()
    self.skip_connection = skip_connection

    self.conv = layers.Conv3D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)
    self.bn = layers.BatchNormalization()
    self.activ = layers.Activation(activation)

  def call(self, inputs, training=False):
    x = inputs
    x = self.conv(x)
    x = self.bn(x, training=training)
    x = self.activ(x)
    if self.skip_connection:
      x = x + inputs
    return x

class ConvTransposeBlock(tf.keras.layers.Layer):
  def __init__(self, filters, kernel_size, strides, padding, activation, skip_connection=False):
    super(ConvTransposeBlock, self).__init__()
    self.skip_connection = skip_connection

    self.conv = layers.Conv3DTranspose(filters, kernel_size, strides=strides, padding=padding, use_bias=False)
    self.bn = layers.BatchNormalization()
    self.activ = layers.Activation(activation)

  def call(self, inputs, training=False):
    x = inputs
    x = self.conv(x)
    x = self.bn(x, training=training)
    x = self.activ(x)
    if self.skip_connection:
      x = x + inputs
    return x


class CNNAutoencoder(tf.keras.Model):
  def __init__(self):
    super(CNNAutoencoder, self).__init__()
    self.encoder = tf.keras.Sequential([
      Input(shape=(64, 64, 64, 1)),
      ConvBlock(filters=8, kernel_size=3, activation=tf.nn.leaky_relu, padding='same', strides=2),
      ConvBlock(filters=8, kernel_size=3, activation=tf.nn.leaky_relu, padding='same', strides=1),
      layers.Dropout(0.2),
      ConvBlock(filters=16, kernel_size=3, activation=tf.nn.leaky_relu, padding='same', strides=2),
      ConvBlock(filters=16, kernel_size=3, activation=tf.nn.leaky_relu, padding='same', strides=1),
      layers.Dropout(0.2),
      ConvBlock(filters=24, kernel_size=3, activation=tf.nn.leaky_relu, padding='same', strides=2),
      ConvBlock(filters=24, kernel_size=3, activation=tf.nn.leaky_relu, padding='same', strides=1),
      layers.Dropout(0.2),
      ConvBlock(filters=32, kernel_size=3, activation=tf.nn.leaky_relu, padding='same', strides=2),
      ConvBlock(filters=32, kernel_size=3, activation=tf.nn.leaky_relu, padding='same', strides=1),
      ConvBlock(filters=4, kernel_size=3, activation=tf.nn.leaky_relu, padding='same', strides=1),
    ])

    self.decoder = tf.keras.Sequential([
      Input(shape=(4, 4, 4, 4)),
      ConvTransposeBlock(filters=32, kernel_size=3, activation=tf.nn.leaky_relu, padding='same', strides=1),
      ConvTransposeBlock(filters=32, kernel_size=3, activation=tf.nn.leaky_relu, padding='same', strides=2),
      layers.Dropout(0.2),
      ConvTransposeBlock(filters=24, kernel_size=3, activation=tf.nn.leaky_relu, padding='same', strides=1),
      ConvTransposeBlock(filters=24, kernel_size=3, activation=tf.nn.leaky_relu, padding='same', strides=2),
      layers.Dropout(0.2),
      ConvTransposeBlock(filters=16, kernel_size=3, activation=tf.nn.leaky_relu, padding='same', strides=1),
      ConvTransposeBlock(filters=16, kernel_size=3, activation=tf.nn.leaky_relu, padding='same', strides=2),
      layers.Dropout(0.2),
      ConvTransposeBlock(filters=8, kernel_size=3, activation=tf.nn.leaky_relu, padding='same', strides=1),
      ConvTransposeBlock(filters=8, kernel_size=3, activation=tf.nn.leaky_relu, padding='same', strides=1),
      layers.Conv3DTranspose(filters=1, kernel_size=3, activation=tf.nn.sigmoid, padding='same', strides=2)
    ])

  def call(self, x):
      encoded = self.encoder(x)
      decoded = self.decoder(encoded)
      return decoded

      

class CustomCheckpoint(Callback):

  def __init__(self, filepath, encoder):
    self.monitor = 'val_loss'
    self.monitor_op = np.less
    self.best = np.Inf

    self.filepath = filepath
    self.encoder = encoder


  def on_epoch_end(self, epoch, logs=None):
    current = logs.get(self.monitor)
    if self.monitor_op(current, self.best):
      self.best = current
      # self.whole.save_weights(self.filepath, overwrite=True)
      self.encoder.save(self.filepath, overwrite=True)  # Whichever you prefer


def train_autoencoder():

    parameters = TrainingArguments()

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    #data_dir = '/mnt/md0/Pycharm_Raid/datasets/abc/tfrecords/64_filled/'
    data_dir = '/mnt/md0/Pycharm_Raid/datasets/abc/tfrecords/128/'
    tfrecords = list(Path(data_dir).rglob('*.tfrecords'))
    dataset = get_abc_dataset(tfrecords)

    validation_dataset = dataset.take(20_000)
    test_dataset = dataset.skip(20_000).take(20_000)
    train_dataset = dataset.skip(40_000)

    validation_dataset = validation_dataset.shuffle(16, reshuffle_each_iteration=True).batch(parameters.batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(parameters.batch_size)
    train_dataset = train_dataset.shuffle(16, reshuffle_each_iteration=True).repeat(100).batch(parameters.batch_size).prefetch(tf.data.AUTOTUNE)

    if train_efficientnet:
      log_dir = "logs/efficient/fit/"
      checkpoint_filepath = './tf_ckpt_efficient/'
    else:
      log_dir = "logs/dcnn/fit/"
      checkpoint_filepath = './tf_ckpt_dcnn/'

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='epoch')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_freq='epoch',
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    strategy = tf.distribute.get_strategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():

        if train_efficientnet:
            autoencoder = efficientnets3d_v2.EfficientAutoencoder(
                input_shape=(128, 128, 128, 1),
                blocks_args=efficientnets3d_v2.RES_128_ARGS,
                top_filters=16,
                final_activation='sigmoid')
        else:
            autoencoder = CNNAutoencoder()


        opt = tf.keras.optimizers.Adam(learning_rate=parameters.learning_rate, epsilon=parameters.adam_eps)
        autoencoder.compile(optimizer=opt, loss=losses.MeanSquaredError())
        autoencoder.build((None, 128, 128, 128, 1))
        autoencoder.summary()
        # autoencoder.encoder.summary()
        # autoencoder.decoder.summary()

    autoencoder.load_weights(checkpoint_filepath)
    autoencoder.fit(
        train_dataset,
        validation_data=validation_dataset,
        verbose=1,
        steps_per_epoch=770_000//parameters.batch_size,
        epochs=parameters.epochs,
        callbacks=[tensorboard_callback, checkpoint_callback])

    autoencoder.evaluate(test_dataset)
    autoencoder.save_weights(log_dir + "/saved_model/weights/autoencoder")
    autoencoder.encoder.save_weights(log_dir + "/saved_model/weights/encoder")
    autoencoder.decoder.save_weights(log_dir + "/saved_model/weights/decoder")


if __name__ == '__main__':
    train_autoencoder()

