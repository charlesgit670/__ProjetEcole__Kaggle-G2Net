from tensorflow.keras import layers,models


class Model:

    def get_model(self, name, load_weights=False):
        match name:
            case "perceptron":
                model = self.__perceptron()
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
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model
