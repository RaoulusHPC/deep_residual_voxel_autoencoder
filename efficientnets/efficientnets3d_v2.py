# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This file is a modification of the official Tensorflow EfficientNetV2 implementation.

import copy
import tensorflow as tf
from efficientnets.layers import DepthwiseConv3D, DepthwiseConv3DTranspose, ExplainableSigmoid
from tensorflow.keras import layers

RES_128_ARGS = [
  {
    "kernel_size": 3,
    "num_repeat": 1,
    "input_filters": 8,
    "output_filters": 16,
    "expand_ratio": 1,
    "se_ratio": 0.25,
    "strides": 2,
    "conv_type": 0,
  }, {
    "kernel_size": 3,
    "num_repeat": 2,
    "input_filters": 16,
    "output_filters": 24,
    "expand_ratio": 4,
    "se_ratio": 0.25,
    "strides": 2,
    "conv_type": 0,
  }, {
    "kernel_size": 3,
    "num_repeat": 3,
    "input_filters": 24,
    "output_filters": 32,
    "expand_ratio": 4,
    "se_ratio": 0.25,
    "strides": 2,
    "conv_type": 0,
  }, {
    "kernel_size": 3,
    "num_repeat": 4,
    "input_filters": 32,
    "output_filters": 40,
    "expand_ratio": 4,
    "se_ratio": 0.25,
    "strides": 2,
    "conv_type": 0,
  }, {
    "kernel_size": 3,
    "num_repeat": 1,
    "input_filters": 40,
    "output_filters": 64,
    "expand_ratio": 4,
    "se_ratio": 0.25,
    "strides": 1,
    "conv_type": 0,
  }
]

RES_64_ARGS = [
  {
    "kernel_size": 3,
    "num_repeat": 2,
    "input_filters": 8,
    "output_filters": 16,
    "expand_ratio": 1,
    "se_ratio": 0.25,
    "strides": 2,
    "conv_type": 0,
  }, {
    "kernel_size": 3,
    "num_repeat": 2,
    "input_filters": 16,
    "output_filters": 24,
    "expand_ratio": 2,
    "se_ratio": 0.25,
    "strides": 2,
    "conv_type": 0,
  }, {
    "kernel_size": 3,
    "num_repeat": 3,
    "input_filters": 24,
    "output_filters": 32,
    "expand_ratio": 2,
    "se_ratio": 0.25,
    "strides": 1,
    "conv_type": 0,
  }, {
    "kernel_size": 3,
    "num_repeat": 4,
    "input_filters": 32,
    "output_filters": 40,
    "expand_ratio": 2,
    "se_ratio": 0.25,
    "strides": 2,
    "conv_type": 0,
  }, {
    "kernel_size": 3,
    "num_repeat": 1,
    "input_filters": 40,
    "output_filters": 64,
    "expand_ratio": 2,
    "se_ratio": 0.25,
    "strides": 1,
    "conv_type": 0,
  }
]

CONV_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 2.0,
        "mode": "fan_out",
        "distribution": "truncated_normal"
    }
}

DENSE_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 1. / 3.,
        "mode": "fan_out",
        "distribution": "uniform"
    }
}

def MBConvBlock(
    input_filters,
    output_filters,
    expand_ratio=1,
    kernel_size=3,
    strides=1,
    se_ratio=0.0,
    bn_momentum=0.9,
    activation="swish",
    survival_probability=0.8,
    explainable_sigmoid=True,
    name=None):
  """MBConv block: Mobile Inverted Residual Bottleneck."""
  bn_axis = -1

  def apply(inputs):
    # Expansion phase
    filters = input_filters * expand_ratio
    if expand_ratio != 1:
      x = layers.Conv3D(
          filters=filters,
          kernel_size=1,
          strides=1,
          kernel_initializer=CONV_KERNEL_INITIALIZER,
          padding="same",
          data_format="channels_last",
          use_bias=False,
          name=name + "expand_conv")(inputs)
      x = layers.BatchNormalization(
          axis=bn_axis,
          momentum=bn_momentum,
          name=name + "expand_bn")(x)
      x = layers.Activation(activation, name=name + "expand_activation")(x)
    else:
      x = inputs

    # Depthwise conv
    x = DepthwiseConv3D(
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        name=name + "dwconv")(x)
    x = layers.BatchNormalization(
        axis=bn_axis, momentum=bn_momentum, name=name + "bn")(x)
    x = layers.Activation(activation, name=name + "activation")(x)

    # Squeeze and excite
    if 0 < se_ratio <= 1:
      filters_se = max(1, int(input_filters * se_ratio))
      se = layers.GlobalAveragePooling3D(name=name + "se_squeeze")(x)

      se_shape = (1, 1, 1, -1)
      se = layers.Reshape(se_shape, name=name + "se_reshape")(se)

      se = layers.Conv3D(
          filters_se,
          1,
          padding="same",
          activation=activation,
          kernel_initializer=CONV_KERNEL_INITIALIZER,
          name=name + "se_reduce")(se)
      se = layers.Conv3D(
          filters,
          1,
          padding='same',
          kernel_initializer=CONV_KERNEL_INITIALIZER,
          name=name + 'se_expand')(se)
      if explainable_sigmoid:
        se = ExplainableSigmoid(name=name + 'se_explainable_sigmoid')(se)
      else:
        se = layers.Activation('sigmoid', name=name + 'se_sigmoid')(se)
      x = layers.multiply([x, se], name=name + 'se_excite')
      
    # Output phase
    x = layers.Conv3D(
        filters=output_filters,
        kernel_size=1,
        strides=1,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        padding="same",
        data_format="channels_last",
        use_bias=False,
        name=name + "project_conv")(x)
    x = layers.BatchNormalization(
        axis=bn_axis, momentum=bn_momentum, name=name + "project_bn")(x)

    if strides == 1 and input_filters == output_filters:
      if survival_probability:
        x = layers.Dropout(
            survival_probability,
            noise_shape=(None, 1, 1, 1, 1),
            name=name + "drop")(x)
      x = layers.add([x, inputs], name=name + "add")
    return x

  return apply


def FusedMBConvBlock(
    input_filters,
    output_filters,
    expand_ratio=1,
    kernel_size=3,
    strides=1,
    se_ratio=0.0,
    bn_momentum=0.9,
    activation="swish",
    survival_probability=0.8,
    explainable_sigmoid=True,
    name=None):
  """Fused MBConv Block: Fusing the proj conv1x1 and depthwise_conv into a conv2d."""
  bn_axis = -1

  def apply(inputs):
    # Expansion phase
    filters = input_filters * expand_ratio
    if expand_ratio != 1:
      x = layers.Conv3D(
          filters=filters,
          kernel_size=kernel_size,
          strides=strides,
          kernel_initializer=CONV_KERNEL_INITIALIZER,
          padding="same",
          data_format="channels_last",
          use_bias=False,
          name=name + "expand_conv")(inputs)
      x = layers.BatchNormalization(
          axis=bn_axis,
          momentum=bn_momentum,
          name=name + "expand_bn")(x)
      x = layers.Activation(activation, name=name + "expand_activation")(x)
    else:
      x = inputs

    # Squeeze and excite
    if 0 < se_ratio <= 1:
      filters_se = max(1, int(input_filters * se_ratio))
      se = layers.GlobalAveragePooling3D(name=name + "se_squeeze")(x)

      se_shape = (1, 1, 1, -1)
      se = layers.Reshape(se_shape, name=name + "se_reshape")(se)

      se = layers.Conv3D(
          filters_se,
          1,
          padding="same",
          activation=activation,
          kernel_initializer=CONV_KERNEL_INITIALIZER,
          name=name + "se_reduce")(se)
      se = layers.Conv3D(
          filters,
          1,
          padding='same',
          kernel_initializer=CONV_KERNEL_INITIALIZER,
          name=name + 'se_expand')(se)
      if explainable_sigmoid:
        se = ExplainableSigmoid(name=name + 'se_explainable_sigmoid')(se)
      else:
        se = layers.Activation('sigmoid', name=name + 'se_sigmoid')(se)
      x = layers.multiply([x, se], name=name + 'se_excite')
      
    # Output phase
    x = layers.Conv3D(
        filters=output_filters,
        kernel_size=1 if expand_ratio != 1 else kernel_size,
        strides=1,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        padding="same",
        data_format="channels_last",
        use_bias=False,
        name=name + "project_conv")(x)
    x = layers.BatchNormalization(
        axis=bn_axis, momentum=bn_momentum, name=name + "project_bn")(x)
    if expand_ratio == 1:
      x = layers.Activation(activation=activation, name=name + 'project_activation')(x)

    if strides == 1 and input_filters == output_filters:
      if survival_probability:
        x = layers.Dropout(
            survival_probability,
            noise_shape=(None, 1, 1, 1, 1),
            name=name + "drop")(x)
      x = layers.add([x, inputs], name=name + "add")
    return x

  return apply


def MBConvTransposeBlock(
    input_filters,
    output_filters,
    expand_ratio=1,
    kernel_size=3,
    strides=1,
    se_ratio=0.0,
    bn_momentum=0.9,
    activation="swish",
    survival_probability = 0.8,
    explainable_sigmoid=True,
    name=None):
  """MBConv block: Mobile Inverted Residual Bottleneck."""
  bn_axis = -1

  def apply(inputs):
    # Expansion phase
    if expand_ratio != 1:
      filters = output_filters * expand_ratio
      x = layers.Conv3D(
          filters=filters,
          kernel_size=1,
          strides=1,
          kernel_initializer=CONV_KERNEL_INITIALIZER,
          padding="same",
          data_format="channels_last",
          use_bias=False,
          name=name + "expand_conv")(inputs)
      x = layers.BatchNormalization(
          axis=bn_axis,
          momentum=bn_momentum,
          name=name + "expand_bn")(x)
      x = layers.Activation(activation, name=name + "expand_activation")(x)
    else:
      filters = input_filters
      x = inputs

    # Depthwise conv
    x = DepthwiseConv3DTranspose(
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        name=name + "dwconv_transpose")(x)
    x = layers.BatchNormalization(
        axis=bn_axis, momentum=bn_momentum, name=name + "bn")(x)
    x = layers.Activation(activation, name=name + "activation")(x)

    # Squeeze and excite
    if 0 < se_ratio <= 1:
      filters_se = max(1, int(input_filters * se_ratio))
      se = layers.GlobalAveragePooling3D(name=name + "se_squeeze")(x)

      se_shape = (1, 1, 1, -1)
      se = layers.Reshape(se_shape, name=name + "se_reshape")(se)

      se = layers.Conv3D(
          filters_se,
          1,
          padding="same",
          activation=activation,
          kernel_initializer=CONV_KERNEL_INITIALIZER,
          name=name + "se_reduce")(se)
      se = layers.Conv3D(
          filters,
          1,
          padding='same',
          kernel_initializer=CONV_KERNEL_INITIALIZER,
          name=name + 'se_expand')(se)
      if explainable_sigmoid:
        se = ExplainableSigmoid(name=name + 'se_explainable_sigmoid')(se)
      else:
        se = layers.Activation('sigmoid', name=name + 'se_sigmoid')(se)
      x = layers.multiply([x, se], name=name + 'se_excite')
      
    # Output phase
    x = layers.Conv3D(
        filters=output_filters,
        kernel_size=1,
        strides=1,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        padding="same",
        data_format="channels_last",
        use_bias=False,
        name=name + "project_conv")(x)
    x = layers.BatchNormalization(
        axis=bn_axis, momentum=bn_momentum, name=name + "project_bn")(x)

    if strides == 1 and input_filters == output_filters:
      if survival_probability:
        x = layers.Dropout(
            survival_probability,
            noise_shape=(None, 1, 1, 1, 1),
            name=name + "drop")(x)
      x = layers.add([x, inputs], name=name + "add")
    return x

  return apply


def FusedMBConvTransposeBlock(
    input_filters,
    output_filters,
    expand_ratio=1,
    kernel_size=3,
    strides=1,
    se_ratio=0.0,
    bn_momentum=0.9,
    activation="swish",
    survival_probability=0.8,
    explainable_sigmoid=True,
    name=None):
  """Fused MBConv Block: Fusing the proj conv1x1 and depthwise_conv into a conv2d."""
  bn_axis = -1

  def apply(inputs):
    # Expansion phase
    if expand_ratio != 1:
      filters = input_filters * expand_ratio
      x = layers.Conv3DTranspose(
          filters=filters,
          kernel_size=kernel_size,
          strides=strides,
          kernel_initializer=CONV_KERNEL_INITIALIZER,
          padding="same",
          data_format="channels_last",
          use_bias=False,
          name=name + "expand_conv_transpose")(inputs)
      x = layers.BatchNormalization(
          axis=bn_axis,
          momentum=bn_momentum,
          name=name + "expand_bn")(x)
      x = layers.Activation(activation, name=name + "expand_activation")(x)
    else:
      filters = input_filters
      x = inputs

    # Squeeze and excite
    if 0 < se_ratio <= 1:
      filters_se = max(1, int(input_filters * se_ratio))
      se = layers.GlobalAveragePooling3D(name=name + "se_squeeze")(x)

      se_shape = (1, 1, 1, -1)
      se = layers.Reshape(se_shape, name=name + "se_reshape")(se)

      se = layers.Conv3D(
          filters_se,
          1,
          padding="same",
          activation=activation,
          kernel_initializer=CONV_KERNEL_INITIALIZER,
          name=name + "se_reduce")(se)
      se = layers.Conv3D(
          filters,
          1,
          padding='same',
          kernel_initializer=CONV_KERNEL_INITIALIZER,
          name=name + 'se_expand')(se)
      if explainable_sigmoid:
        se = ExplainableSigmoid(name=name + 'se_explainable_sigmoid')(se)
      else:
        se = layers.Activation('sigmoid', name=name + 'se_sigmoid')(se)
      x = layers.multiply([x, se], name=name + 'se_excite')
      
    # Output phase
    x = layers.Conv3D(
        filters=output_filters,
        kernel_size=1 if expand_ratio != 1 else kernel_size,
        strides=1,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        padding="same",
        data_format="channels_last",
        use_bias=False,
        name=name + "project_conv")(x)
    x = layers.BatchNormalization(
        axis=bn_axis, momentum=bn_momentum, name=name + "project_bn")(x)
    if expand_ratio == 1:
      x = layers.Activation(activation=activation, name=name + 'project_activation')(x)

    if strides == 1 and input_filters == output_filters:
      if survival_probability:
        x = layers.Dropout(
            survival_probability,
            noise_shape=(None, 1, 1, 1, 1),
            name=name + "drop")(x)
      x = layers.add([x, inputs], name=name + "add")
    return x

  return apply


def EfficientNet(
    input_shape,
    blocks_args,
    top_filters=128,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    bn_momentum=0.9,
    activation="swish",
    model_name="efficientnet",
    include_top=False,
    pooling=None,
    classes=1,
    classifier_activation=None):

  img_input = layers.Input(shape=input_shape)

  bn_axis = -1

  x = img_input

  # Build stem
  stem_filters = blocks_args[0]["input_filters"]
  x = layers.Conv3D(
      filters=stem_filters,
      kernel_size=3,
      strides=2,
      kernel_initializer=CONV_KERNEL_INITIALIZER,
      padding="same",
      use_bias=False,
      name="stem_conv")(x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=bn_momentum,
      name="stem_bn")(x)
  x = layers.Activation(activation, name="stem_activation")(x)

  # Build blocks
  blocks_args = copy.deepcopy(blocks_args)
  b = 0
  blocks = float(sum(args["num_repeat"] for args in blocks_args))


  for (i, args) in enumerate(blocks_args):
    assert args["num_repeat"] > 0

    # Determine which conv type to use:
    block = {0: MBConvBlock, 1: FusedMBConvBlock}[args.pop("conv_type")]
    repeats = args.pop("num_repeat")
    for j in range(repeats):
      # The first block needs to take care of stride and filter size increase.
      if j > 0:
        args["strides"] = 1
        args["input_filters"] = args["output_filters"]

      x = block(
          activation=activation,
          bn_momentum=bn_momentum,
          survival_probability=drop_connect_rate * b / blocks,
          name="block{}{}_".format(i + 1, chr(j + 97)),
          **args)(x)
      b += 1

  # Build top
  x = layers.Conv3D(
      filters=top_filters,
      kernel_size=1,
      strides=1,
      kernel_initializer=CONV_KERNEL_INITIALIZER,
      padding="same",
      data_format="channels_last",
      use_bias=False,
      name="top_conv")(x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=bn_momentum,
      name="top_bn")(x)
  x = layers.Activation(activation=activation, name="top_activation")(x)

  if include_top:
    x = layers.GlobalAveragePooling3D(name="avg_pool")(x)
    if dropout_rate > 0:
      x = layers.Dropout(dropout_rate, name="top_dropout")(x)
    x = layers.Dense(
        classes,
        activation=classifier_activation,
        kernel_initializer=DENSE_KERNEL_INITIALIZER,
        bias_initializer=tf.constant_initializer(0),
        name="predictions")(x)
  else:
    if pooling == "avg":
      x = layers.GlobalAveragePooling3D(name="avg_pool")(x)
    elif pooling == "max":
      x = layers.GlobalMaxPooling3D(name="max_pool")(x)

  inputs = img_input

  # Create model.
  model = tf.keras.Model(inputs, x, name=model_name)

  return model


def EfficientNetDecoder(
    input_shape,
    blocks_args,
    drop_connect_rate=0.2,
    bn_momentum=0.9,
    activation="swish",
    model_name="efficientnet_decoder",
    final_activation=None):

  img_input = layers.Input(shape=input_shape)

  bn_axis = -1

  x = img_input

  x = layers.Reshape((4, 4, 4, -1), name="top_reshape")(x)

  x = layers.Conv3D(
      filters=blocks_args[-1]["output_filters"],
      kernel_size=1,
      strides=1,
      kernel_initializer=CONV_KERNEL_INITIALIZER,
      padding="same",
      data_format="channels_last",
      use_bias=False,
      name="top_conv")(x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=bn_momentum,
      name="top_bn")(x)
  x = layers.Activation(activation=activation, name="top_activation")(x)

  # Build blocks
  blocks_args = copy.deepcopy(blocks_args)
  b = 0
  blocks = float(sum(args["num_repeat"] for args in blocks_args))

  for (i, args) in enumerate(blocks_args[::-1]):
    assert args["num_repeat"] > 0

    strides = args["strides"]
    input_filters = args["input_filters"]

    block = {0: MBConvTransposeBlock, 1: FusedMBConvTransposeBlock}[args.pop("conv_type")]
    repeats = args.pop("num_repeat")
    for j in range(repeats):
      # The last block needs to take care of stride and filter size increase.
      if j < repeats - 1:
        args["strides"] = 1
        args["input_filters"] = args["output_filters"]
      else:
        args["strides"] = strides
        args["input_filters"] = args["output_filters"]
        args["output_filters"] = input_filters

      x = block(
          activation=activation,
          bn_momentum=bn_momentum,
          survival_probability=drop_connect_rate * (blocks-b-1) / blocks,
          name="block{}{}_".format(len(blocks_args) - i, chr(repeats - j - 1 + 97)),
          **args)(x)
      b += 1

  # Build stem
  x = layers.Conv3DTranspose(
      filters=1,
      kernel_size=3,
      strides=2,
      kernel_initializer=CONV_KERNEL_INITIALIZER,
      padding="same",
      use_bias=True,
      name="stem_conv_transpose")(x)
  x = layers.Activation(final_activation, dtype="float32", name="final_activation")(x)

  inputs = img_input

  # Create model.
  model = tf.keras.Model(inputs, x, name=model_name)

  return model


class EfficientAutoencoder(tf.keras.Model):
  def __init__(self, input_shape, blocks_args, top_filters, final_activation):
    super(EfficientAutoencoder, self).__init__()
    self.encoder = EfficientNet(
      input_shape=input_shape,
      blocks_args=blocks_args,
      top_filters=top_filters)

    self.decoder = EfficientNetDecoder(
      input_shape=self.encoder.output_shape[1:],
      blocks_args=blocks_args,
      final_activation=final_activation)

  def call(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

class EfficientAutoencoder2(tf.keras.Model):
  def __init__(self, input_shape, blocks_args, top_filters, latent_dim, final_activation):
    super(EfficientAutoencoder2, self).__init__()
    self.encoder = EfficientNet(
      input_shape=input_shape,
      blocks_args=blocks_args,
      top_filters=top_filters,
      include_top=True,
      classes=latent_dim,
      classifier_activation='swish')

    self.decoder = EfficientNetDecoder(
      input_shape=self.encoder.output_shape[1:],
      blocks_args=blocks_args,
      final_activation=final_activation)

  def call(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x


class EfficentClassifier(tf.keras.Model):
  def __init__(self, input_shape, blocks_args, top_filters, classes, final_activation):
    super(EfficentClassifier, self).__init__()

    self.encoder = EfficientNet(
      input_shape=input_shape,
      blocks_args=blocks_args,
      top_filters=top_filters,
      include_top=True,
      classes=64,
      classifier_activation=tf.keras.activations.swish)
    self.dense = layers.Dense(classes, activation=final_activation, activity_regularizer=tf.keras.regularizers.L1L2(1e-5), dtype='float32')

  def call(self, x):
    x = self.encoder(x)
    x = self.dense(x)
    return x


if __name__ == "__main__":
  # model = EfficientNet(
  #   input_shape=(128, 128, 128, 1),
  #   blocks_args=RES_128_ARGS,
  #   top_filters=128)

  model = EfficientAutoencoder(
    input_shape=(128, 128, 128, 1),
    blocks_args=RES_128_ARGS,
    top_filters=256,
    latent_dim=64,
    final_activation='sigmoid')
  model.build((None, 128, 128, 128, 1))
  model.summary(line_length=125)
  model.encoder.summary(line_length=125)
  model.decoder.summary(line_length=125)
