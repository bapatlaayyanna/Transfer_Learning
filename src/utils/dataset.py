import tensorflow as tf
import numpy as np
from src.utils.common import read_config

def get_dataset():
    (x_train_full,y_train_full),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
    x_train_full = x_train_full/255.
    y_train_full = y_train_full/255.
    x_test = x_test/255.
    x_valid , x_train = x_train_full[:5000], x_train_full[5000:]
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    return x_train,y_train,x_valid,y_valid,x_test,y_test

def setting_seed(seed_value):
    tf.random.set_seed(seed_value)
    np.random.seed(seed_value)

def get_layers():
    LAYERS = [
        tf.keras.layers.Flatten(input_shape = [28,28],name = "InputLayer"),
        tf.keras.layers.Dense(300,activation = "LeakyReLU", name = "HiddenLayer1"),
        tf.keras.layers.Dense(100,activation = "LeakyReLU", name = "HiddenLayer2"),
        tf.keras.layers.Dense(10, activation = "softmax", name = "OutputLayer")
    ]
    return LAYERS








