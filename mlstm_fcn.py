from attention import AttentionLSTMCell, RNN
from cbam import cbam
import tensorflow as tf


def conv_block(inputs):
    x = tf.keras.layers.SeparableConv1D(128,8,padding="same",kernel_initializer="he_normal")(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=0.99)(x)
    x = tf.keras.layers.ELU()(x)
    x = cbam(x)
    x = tf.keras.layers.SeparableConv1D(256,5,padding="same",kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.99)(x)
    x = tf.keras.layers.ELU()(x)
    x = cbam(x)
    x = tf.keras.layers.SeparableConv1D(128,3,padding="same",kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.99)(x)
    x =  tf.keras.layers.ELU()(x)
    return tf.keras.layers.GlobalAvgPool1D()(x)


def lstm_block(inputs):
    x = tf.keras.layers.Permute((2, 1))(inputs)
    x = tf.keras.layers.GRU(128, return_sequences=True)(x)
    cell = AttentionLSTMCell(128, dropout=0.2, recurrent_dropout=0.2)
    return RNN(cell)(x)


def mlstm_fcn(inputs):
    x1 = conv_block(inputs)
    x2 = lstm_block(inputs)
    return tf.keras.layers.Concatenate()([x1,x2])