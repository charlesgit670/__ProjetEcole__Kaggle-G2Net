from tensorflow.keras import layers,models


class Model:

    def get_model(self, name):
        match name:
            case "perceptron":
                return self.perceptron()
            case default:
                raise Exception(name+" : This model is not supported")

    def perceptron(self):
        model = models.Sequential([
            layers.Flatten(input_shape=(360,4096,1)),
            layers.Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model
