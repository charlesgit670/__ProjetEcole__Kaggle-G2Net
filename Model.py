from tensorflow.keras import layers,models
import tensorflow as tf


class Model:

    def get_model(self, name, load_weights=False):
        match name:
            case "perceptron":
                model = self.__perceptron()
            case "efficientNet7":
                model = self.__efficientNet7()
            case "conv_layer":
                model = self.__conv_layer()
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
            layers.Flatten(),
            layers.Dense(1, activation="sigmoid") #0.05
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC', name='ROC'), tf.keras.metrics.AUC(curve='PR', name='PR')])

        return model

    def __efficientNet7(self):

        input = layers.Input(shape=((360,128)))
        # in_conv = tf.keras.layers.Conv2D(3, 7, strides=(1, 1), padding='same')
        x = tf.expand_dims(input, axis=-1)
        x = tf.tile(x, [1, 1, 1, 3])
        x = tf.cast(x, tf.float32)
        base = tf.keras.applications.efficientnet_v2.EfficientNetV2S(input_shape=(360,128, 3),
                                                                        include_top=False,
                                                                        weights='imagenet')
        # x = in_conv(input)
        x = base(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.30)(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs=input, outputs=x)
        optimizer = tf.optimizers.Adam(learning_rate=4e-4)
        model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC', name='ROC'), tf.keras.metrics.AUC(curve='PR', name='PR')])

        return model

    def __conv_layer(self):
        model = models.Sequential([
            layers.Conv2D(32, kernel_size=3, activation='relu', padding="same", input_shape=(360,128,1)),
            layers.Conv2D(32, kernel_size=3, activation='relu', padding="same"),
            layers.MaxPool2D(),
            layers.Dropout(0.3),

            layers.Conv2D(64, kernel_size=3, activation='relu', padding="same"),
            layers.Conv2D(64, kernel_size=3, activation='relu', padding="same"),
            layers.MaxPool2D(),
            layers.Dropout(0.3),

            layers.Conv2D(128, kernel_size=3, activation='relu', padding="same"),
            layers.Conv2D(128, kernel_size=3, activation='relu', padding="same"),
            layers.MaxPool2D(),
            layers.Dropout(0.3),

            layers.Conv2D(256, kernel_size=3, activation='relu', padding="same"),
            layers.Conv2D(256, kernel_size=3, activation='relu', padding="same"),
            layers.MaxPool2D(),
            layers.Dropout(0.3),

            layers.Conv2D(512, kernel_size=3, activation='relu', padding="same"),
            layers.Conv2D(512, kernel_size=3, activation='relu', padding="same"),
            layers.MaxPool2D(),
            layers.Dropout(0.3),

            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC', name='ROC'),
                               tf.keras.metrics.AUC(curve='PR', name='PR')])

        return model




