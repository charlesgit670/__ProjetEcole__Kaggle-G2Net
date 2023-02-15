import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from Model import Model
from DataGenerator import DataGenerator
from DataProcessedGenerator import DataProcessedGenerator

def L1_H1_mean_convert_output(output, BATCH_SIZE):
    rest = len(output) % (BATCH_SIZE * 2)
    output_final = []
    for i in range(0, len(output) - rest, BATCH_SIZE * 2):
        for j in range(i, i + BATCH_SIZE):
            output_final.append((output[j] + output[j + BATCH_SIZE]) / 2)

    for i in range(len(output)-rest, len(output)-rest//2):
        output_final.append((output[i] + output[i + rest//2]) / 2)

    return np.array(output_final)


if __name__ == '__main__':
    BATCH_SIZE = 8
    MODEL_NAME = "efficientNet7"
    L1_H1_mean = False # moyenne de L1 et H1 en sortie du modèle si True
    CHANNEL = 1 if L1_H1_mean else 2

    label_file = pd.read_csv("data/train_labels.csv")
    label_file = label_file[label_file["target"] >= 0]

    label_file_train, label_file_val = train_test_split(label_file, test_size=0.2, random_state=42)

    train_gen = DataProcessedGenerator("train_processed", label_file_train, BATCH_SIZE, L1_H1_mean=L1_H1_mean)
    val_gen = DataProcessedGenerator("train_processed", label_file_val, BATCH_SIZE, L1_H1_mean=L1_H1_mean)

    object = Model()
    model = object.get_model(MODEL_NAME, False, CHANNEL)

    # model.summary()
    EPOCHS = 20
    checkpoint_filepath = "model_weights/" + MODEL_NAME + "/" + MODEL_NAME
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_ROC',
        mode='max',
        save_best_only=True)
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[tf.keras.callbacks.TensorBoard("logs/" + MODEL_NAME),
                                                                            model_checkpoint_callback])

    output_train = model.predict(train_gen)
    output_test = model.predict(val_gen)

    if L1_H1_mean:
        output_train = L1_H1_mean_convert_output(output_train, BATCH_SIZE)
        output_test = L1_H1_mean_convert_output(output_test, BATCH_SIZE)

    for i in range(1,100):
        threshold = i*0.01
        output_train_predict = np.where(output_train > threshold, 1, 0)
        output_test_predict = np.where(output_test > threshold, 1, 0)
        cm_train = tf.math.confusion_matrix(label_file_train["target"],output_train_predict)
        cm_test = tf.math.confusion_matrix(label_file_val["target"], output_test_predict)
        print("threshold : ",threshold)
        print(cm_train)
        print(cm_test)

    plt.figure(figsize=(15, 8))
    plt.title('Predicted Target Distribution train')
    pd.DataFrame(output_train, columns = ["target"])["target"].plot(kind='hist', bins=32)
    plt.xlabel('Count')
    plt.xlabel('Predicted Target')
    plt.xlim(0, 1)
    plt.grid()
    plt.show()

    plt.figure(figsize=(15, 8))
    plt.title('Predicted Target Distribution test')
    pd.DataFrame(output_test, columns = ["target"])["target"].plot(kind='hist', bins=32)
    plt.xlabel('Count')
    plt.xlabel('Predicted Target')
    plt.xlim(0, 1)
    plt.grid()
    plt.show()