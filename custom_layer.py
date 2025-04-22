from tensorflow.keras.layers import Layer
import tensorflow as tf

class CustomScaleLayer(Layer):
    def __init__(self, scale=1.0, **kwargs):
        super(CustomScaleLayer, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        if isinstance(inputs, list):
            inputs = tf.add_n(inputs)  # Combine input tensors, or use tf.concat() if needed
        return tf.multiply(inputs, self.scale)

    def get_config(self):
        config = super(CustomScaleLayer, self).get_config()
        config.update({'scale': self.scale})
        return config
