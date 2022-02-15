import tensorflow as tf
from tensorflow.keras import layers

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}

class DepthwiseConv3D(layers.Layer):

  def __init__(self, kernel_size, strides, padding, **kwargs):
    super(DepthwiseConv3D, self).__init__(**kwargs)
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.use_gpu = True

  def build(self, input_shape):
    self.channels = input_shape[-1]
    self.filters = self.add_weight(
      shape=(self.kernel_size, self.kernel_size, self.kernel_size, 1, self.channels),
      initializer=CONV_KERNEL_INITIALIZER,
      trainable=True,
      name='conv_weight'
    )

  def get_config(self):
      config = super().get_config().copy()
      config.update({
          'kernel_size': self.kernel_size,
          'strides': self.strides,
          'padding': self.padding
      })
      return config

  def call(self, x):
      if tf.config.list_physical_devices('GPU'):
          strides = [1, self.strides, self.strides, self.strides, 1]
          y = tf.nn.conv3d(x, self.filters, strides=strides, padding=self.padding.upper())
      else:
          strides = [1, self.strides, self.strides, self.strides, 1]
          y = tf.concat(
            [tf.nn.conv3d(
              x[:, :, :, :, i:i+1],
              self.filters[:, :, :, :, i:i+1],
              strides=strides,
              padding=self.padding.upper()) for i in range(self.channels)],
            axis=-1)
      return y


class DepthwiseConv3DTranspose(layers.Layer):

  def __init__(self, kernel_size, strides, padding, **kwargs):
        super(DepthwiseConv3DTranspose, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

  def build(self, input_shape):

      self.channels = input_shape[-1]

      self.filters = self.add_weight(
          shape=(self.kernel_size, self.kernel_size, self.kernel_size, 1, self.channels),
          initializer=CONV_KERNEL_INITIALIZER,
          trainable=True,
          name='conv_transpose_weight'
      )

  def get_config(self):
      config = super().get_config().copy()
      config.update({
          'kernel_size': self.kernel_size,
          'strides': self.strides,
          'padding': self.padding,
      })
      return config

  def call(self, x):
      if tf.config.list_physical_devices('GPU'):
        output_shape = [tf.shape(x)[0], self.strides * x.shape[1], self.strides * x.shape[2], self.strides * x.shape[3], x.shape[4]]
        y = tf.nn.conv3d_transpose(x, self.filters, output_shape=output_shape, strides=self.strides, padding=self.padding.upper())
      else:
        output_shape = [tf.shape(x)[0], self.strides * x.shape[1], self.strides * x.shape[2], self.strides * x.shape[3], 1]
        y = tf.concat(
            [tf.nn.conv3d_transpose(
              x[:, :, :, :, i:i+1],
              self.filters[:, :, :, :, i:i+1],
              output_shape=output_shape,
              strides=self.strides,
              padding=self.padding.upper()) for i in range(self.channels)],
            axis=-1)
      return y

# Separate sigmoid into tanh followed and a linear operation, which enables gradient based epsilon-LRP
class ExplainableSigmoid(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(ExplainableSigmoid, self).__init__(**kwargs)
    def scaled_tanh(x):
      return tf.nn.tanh(0.5*x)
    self.activation = scaled_tanh

  def call(self, x):
    x = self.activation(x)
    x = (x + 1) / 2
    return x
