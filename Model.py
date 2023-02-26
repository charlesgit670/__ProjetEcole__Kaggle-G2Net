from tensorflow.keras import layers,models
import tensorflow as tf


class Model:
    def __init__(self):
        self.__data_augmentation = tf.keras.Sequential([
                                        layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
                                ])
    def get_model(self, name, load_weights=False, channel=1):
        match name:
            case "perceptron":
                model = self.__perceptron()
            case "efficientNetV2S":
                model = self.__efficientNetV2S(channel)
            case default:
                raise Exception(name+" : This model is not supported")
        if load_weights:
            try:
                model.load_weights("model_weights/"+name+"/"+name)
            except Exception:
                raise Exception("Error while trying to load weights for the model : /model_weights/"+name+"/"+name)
        return model

    def __perceptron(self):
        model = models.Sequential([
            # self.__data_augmentation,
            layers.Flatten(),
            layers.Dense(1, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.L2(0.01), kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1))
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC', name='ROC'), tf.keras.metrics.AUC(curve='PR', name='PR')])

        return model

    def __efficientNetV2S(self, channel):
        input = layers.Input(shape=((360, 128,channel)))
        # x = self.__data_augmentation(input)
        x = layers.Conv2D(3, 3, padding='same', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1))(input)
        x = tf.keras.applications.efficientnet_v2.EfficientNetV2S(input_shape=(360,128,3),
                                                                        include_top=False,
                                                                        weights='imagenet'
                                                                        )(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1))(x)

        model = tf.keras.Model(inputs=input, outputs=output)

        lr_decay = tf.keras.optimizers.schedules.ExponentialDecay(
            0.0001, 2500, 0.01, staircase=False, name=None
        )
        adam = tf.keras.optimizers.Adam(lr_decay)
        model.compile(optimizer=adam, loss='binary_crossentropy',metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC', name='ROC'), tf.keras.metrics.AUC(curve='PR', name='PR')])

        return model






