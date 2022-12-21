from tensorflow.keras import layers,models
import tensorflow as tf


class Model:

    def get_model(self, name, load_weights=False):
        match name:
            case "perceptron":
                model = self.__perceptron()
            case "efficientNet7":
                model = self.__efficientNet7()
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
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')])

        return model

    def __efficientNet7(self):

        input = layers.Input(shape=((360,128, 1)))
        in_conv = tf.keras.layers.Conv2D(3, 7, strides=(1, 1), padding='same')
        base = tf.keras.applications.efficientnet_v2.EfficientNetV2S(input_shape=(360,128, 3),
                                                                        include_top=False,
                                                                        weights=None)
        x = in_conv(input)
        x = base(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs=input, outputs=x)

        model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC', name='ROC'), tf.keras.metrics.AUC(curve='PR', name='PR')])

        return model