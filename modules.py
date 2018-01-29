import tensorflow as tf


class TwoLayerModule:
    def __init__(self, input_dim, output_dim, hidden_dim = 128):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.initializer = tf.contrib.layers.xavier_initializer()

    def __call__(self, tf_in):
        tf_out = tf.layers.dense(tf_in, self.hidden_dim, activation=tf.nn.relu, kernel_initializer=self.initializer)
        tf_out = tf.layers.dense(tf_out, self.output_dim, activation=tf.nn.relu, kernel_initializer=self.initializer)
        return tf_out