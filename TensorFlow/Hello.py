import tensorflow as tf


def prepare_minist_features_and_labels(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y


def mnist_dataset():
    (x, y), _ = tf.keras.datasets.mnist.load_data()
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_minist_features_and_labels)
    ds = ds.take(20000).shuffle(20000).batch(100)
    return ds


train_dataset = mnist_dataset()
