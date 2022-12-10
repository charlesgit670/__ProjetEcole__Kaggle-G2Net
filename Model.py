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
            layers.Flatten(input_shape=(360,4096,1)),
            layers.Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')])

        return model

    def __efficientNet7(self):
        model = tf.keras.applications.efficientnet.EfficientNetB7(input_shape=(360,128,1),
                                                                        include_top=True,
                                                                        classes=1,
                                                                        weights=None,
                                                                        classifier_activation='sigmoid')
        model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy', tf.keras.metrics.AUC(curve='PR')])
        return model