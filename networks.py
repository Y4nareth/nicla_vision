# networks.py>

import tensorflow as tf

num_classes = 2

classes = ['Person','Background']

class Networks:
    def tiny_darknet():
        """
        Darknet model tiny version -> https://pjreddie.com/darknet/tiny-darknet/
        """
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Input(shape=(416, 416, 3)))
    
        # Layer 0
        model.add(tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))

        # Layer 1
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        # Layer 2
        model.add(tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))

        # Layer 3
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        # Layer 4
        model.add(tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))

        # Layer 5
        model.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))

        # Layer 6
        model.add(tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))

        # Layer 7
        model.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))

        # Layer 8
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        # Layer 9
        model.add(tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))

        # Layer 10
        model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))

        # Layer 11
        model.add(tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))

        # Layer 12
        model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))

        # Layer 13
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        # Layer 14
        model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))

        # Layer 15
        model.add(tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))

        # Layer 16
        model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))

        # Layer 17
        model.add(tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))

        # Layer 18
        model.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))

        # Layer 19
        model.add(tf.keras.layers.Conv2D(1000, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))

        # Layer 20
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(4096, activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
        model.add(tf.keras.layers.Dense(1470, activation='linear'))

        return model


    