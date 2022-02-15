import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import numpy as np
from tensorflow.keras import losses
import os
import pyvista as pv
from pathlib import Path
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
############################
from efficientnets import efficientnets3d_v2
from train_autoencoder import get_abc_dataset

tf.compat.v1.enable_eager_execution()

def get_labeled_dataset(tfrecords, batch_size=1, repeat=1, augment_function=None):
    raw_component_dataset = tf.data.TFRecordDataset(tfrecords, compression_type='GZIP')

    # Create a dictionary describing the features.
    component_feature_description = {
        'component_raw': tf.io.FixedLenFeature([], tf.string),
        'label_raw': tf.io.FixedLenFeature([], tf.string),
    }

    def read_tfrecord(serialized_example):
        # Parse the input tf.train.Example proto using the dictionary above.
        example = tf.io.parse_single_example(serialized_example, component_feature_description)
        component = tf.io.parse_tensor(example['component_raw'], out_type=bool)
        label = tf.io.parse_tensor(example['label_raw'], out_type=float)
        label = label[0]
        component = tf.cast(component, 'float32')
        component = tf.expand_dims(component, axis=-1)
        return component, label

    dataset = raw_component_dataset.map(read_tfrecord)
    dataset = dataset.shuffle(1024, reshuffle_each_iteration=True)
    dataset = dataset.repeat(repeat)
    if augment_function:
        dataset = dataset.map(augment_function)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
cwd = os.getcwd()
nr_parts = 3

autoencoder = efficientnets3d_v2.EfficientAutoencoder(
    input_shape=(128, 128, 128, 1),
    blocks_args=efficientnets3d_v2.RES_128_ARGS,
    top_filters=16,
    final_activation='sigmoid')

# autoencoder.decoder = tf.keras.models.load_model('tensorboards/_tensorboard_8filters/onlyEncoder/encoderweights.tf/')
# autoencoder.encoder= tf.keras.models.load_model('tensorboards/_tensorboard_8filters/onlyEncoder/encoderweights.tf/')
checkpoint_filepath = './tf_ckpt_efficient/'

# opt = keras.optimizers.Adam(lr=0.01, epsilon=1e-5)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder.build((None, 128, 128, 128, 1))
autoencoder.load_weights(checkpoint_filepath)
autoencoder.summary()

take_abc_data = False
if take_abc_data:
    data_dir = '/mnt/md0/Pycharm_Raid/datasets/abc/tfrecords/128/'
    tfrecords = list(Path(data_dir).rglob('*.tfrecords'))
    dataset = get_abc_dataset(tfrecords)

    validation_dataset = dataset.take(20000)
    test_dataset = dataset.skip(20000).take(20000)
    train_dataset = dataset.skip(40000)

    test_dataset = test_dataset.take(200)
    test_dataset = test_dataset.shuffle(64, reshuffle_each_iteration=True)
else:
    tfrecords = ['/mnt/md0/Pycharm_Raid/datasets/labeled_data/labeled_data_128.tfrecords']
    test_dataset = get_labeled_dataset(tfrecords, batch_size=16, repeat=1)
#
test_batch = np.zeros((nr_parts, 128, 128, 128, 1))
# for i in range(nr_parts):
#   X[i, :, :, :] = np.load(os.path.join(CAD_Files_Path, filenames[randint(0, 9600)]))
while True:
    if take_abc_data:
        i = 0
        for images in test_dataset.take(nr_parts):
            test_batch[i, :, :, :, :] = images[0].numpy()
            i += 1

        output = autoencoder.predict(test_batch)  # X.reshape((nr_parts, 128, 128, 128, 1))

        p = pv.Plotter(shape=(nr_parts, 2))

        for row in range(nr_parts):
            for column in range(2):
                if (column == 0):
                    p.subplot(row, column)
                    p.add_volume(test_batch[row].reshape((128, 128, 128)), cmap="viridis", shade=True)  #
                    p.add_text("Input" + str(row))
                elif (column == 1):
                    p.subplot(row, column)
                    p.add_volume(output[row].reshape((128, 128, 128)), cmap="viridis", shade=True)
                    p.add_text("Output" + str(row))

        p.link_views()
        p.show()

    else:
        for data, label in test_dataset:
            output = autoencoder.predict(test_batch)  # X.reshape((nr_parts, 128, 128, 128, 1))

            p = pv.Plotter(shape=(nr_parts, 2))

            for row in range(nr_parts):
                for column in range(2):
                    if (column == 0):
                        p.subplot(row, column)
                        p.add_volume(test_batch[row].reshape((128, 128, 128)), cmap="viridis", shade=True)  #
                        p.add_text("Input" + str(row))
                    elif (column == 1):
                        p.subplot(row, column)
                        p.add_volume(output[row].reshape((128, 128, 128)), cmap="viridis", shade=True)
                        p.add_text("Output" + str(row))

            p.link_views()
            p.show()
