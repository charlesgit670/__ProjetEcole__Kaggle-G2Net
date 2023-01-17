import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from Model import Model
from DataGenerator import DataGenerator
from DataProcessedGenerator import DataProcessedGenerator
from training import L1_H1_mean_convert_output


if __name__ == '__main__':
    MODEL_NAME = "efficientNet7"
    BATCH_SIZE = 32
    L1_H1_mean = False
    CHANNEL = 1 if L1_H1_mean else 2

    test_file = pd.read_csv("data/sample_submission.csv")
    test_gen = DataProcessedGenerator("test_clean", test_file, BATCH_SIZE, L1_H1_mean=L1_H1_mean)
    object = Model()
    model = object.get_model(MODEL_NAME, True, CHANNEL)

    output = model.predict(test_gen)

    if L1_H1_mean:
        output = L1_H1_mean_convert_output(output)

    test_file["target"] = output
    test_file.to_csv('submission.csv', index=False)

    # Sanity check, predicted target distribution
    plt.figure(figsize=(15, 8))
    plt.title('Predicted Target Distribution')
    test_file['target'].plot(kind='hist', bins=32)
    plt.xlabel('Count')
    plt.xlabel('Predicted Target')
    plt.xlim(0, 1)
    plt.grid()
    plt.show()















