import pandas as pd
import numpy as np
from Model import Model
from CustomDataGen import CustomDataGen



if __name__ == '__main__':
    MODEL_NAME = "perceptron"
    BATCH_SIZE = 64

    test_file = pd.read_csv("data/sample_submission.csv")
    test_gen = CustomDataGen("test", test_file, BATCH_SIZE)
    object = Model()
    model_perceptron = object.get_model(MODEL_NAME, True)

    output = model_perceptron.predict(test_gen)

    test_file["target"] = output
    test_file.to_csv('submission.csv', index=False)















