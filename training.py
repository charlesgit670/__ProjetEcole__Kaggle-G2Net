import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

from Model import Model
from CustomDataGen import CustomDataGen

if __name__ == '__main__':
    BATCH_SIZE = 64
    MODEL_NAME = "perceptron"

    label_file = pd.read_csv("data/train_labels.csv")
    label_file_train, label_file_val = train_test_split(label_file, test_size=0.2, random_state=42)

    train_gen = CustomDataGen("train", label_file_train, BATCH_SIZE)
    val_gen = CustomDataGen("train", label_file_val, BATCH_SIZE)

    object = Model()
    model_perceptron = object.get_model(MODEL_NAME, False)

    model_perceptron.summary()

    model_perceptron.fit(train_gen, validation_data=val_gen, epochs=2,
                         callbacks=[tf.keras.callbacks.TensorBoard("logs/"+MODEL_NAME)], )
    model_perceptron.save_weights("model_weights/"+MODEL_NAME+"/"+MODEL_NAME)