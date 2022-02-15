
import tensorflow as tf
from tensorflow.keras import losses

from pathlib import Path

from efficientnets import efficientnets3d_v2
from dataclasses import dataclass

@dataclass
class TrainingArguments:
    epochs: int = 16
    batch_size: int = 16
    learning_rate: float = 2e-4
    adam_eps: float = 1e-6
    attribute_to_train_for = 0
    use_memory_growth: bool = True


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


parameters = TrainingArguments()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            if parameters.use_memory_growth:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

tfrecords = ['/mnt/md0/Pycharm_Raid/datasets/labeled_data/labeled_data_128.tfrecords']
train_dataset = get_labeled_dataset(tfrecords, batch_size=parameters.batch_size, repeat=1)
train_dataset = train_dataset.take(10)

for data, label in train_dataset:
    print(data.shape, label.shape)
    break

classifier = efficientnets3d_v2.EfficentClassifier(
    input_shape=(128, 128, 128, 1),
    blocks_args=efficientnets3d_v2.RES_128_ARGS,
    top_filters=16,
    classes=11,
    final_activation='sigmoid')
classifier.build((None, 128, 128, 128, 1))

#classifier.encoder.trainable = False
classifier.summary()

#classifier.encoder.load_weights('/home/rgs/PycharmProjects/AutoEncoder/logs/efficient/fit/saved_model/weights/encoder')

classifier.encoder.trainable = False
opt = tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon=parameters.adam_eps)
classifier.compile(optimizer=opt, loss=losses.MeanSquaredError(), metrics=[tf.metrics.MeanSquaredError(), tf.metrics.CategoricalAccuracy()])
classifier.fit(train_dataset,
               epochs=16,
               )

if True:
    classifier.encoder.trainable = True
    opt = tf.keras.optimizers.Adam(learning_rate=parameters.learning_rate, epsilon=parameters.adam_eps)
    classifier.compile(optimizer=opt, loss=losses.MeanSquaredError(), metrics=[tf.metrics.MeanSquaredError(), tf.metrics.CategoricalAccuracy()])

    classifier.fit(train_dataset,
                   epochs=parameters.epochs,
                   )

classifier.save_weights("./weights/128/classifier/")
#classifier.save('./model/effnet.h5', save_format='tf')


# lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
#     initial_learning_rate=1e-3, decay_steps=(183 // parameters.batch_size) * parameters.epochs, alpha=0.01, name=None
# )