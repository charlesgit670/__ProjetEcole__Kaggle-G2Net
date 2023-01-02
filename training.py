import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import math

from Model import Model
from CustomDataGen import CustomDataGen

class WeightDecayCallback(tf.keras.callbacks.Callback):
    def __init__(self, wd_ratio=0.01):
        self.step_counter = 0
        self.wd_ratio = wd_ratio

    def on_epoch_begin(self, epoch, logs=None):
        model.optimizer.weight_decay = model.optimizer.learning_rate * self.wd_ratio
        print(
            f'learning rate: {model.optimizer.learning_rate.numpy():.2e}, weight decay: {model.optimizer.weight_decay.numpy():.2e}')



# Cosine Decay with exponential warmup
def lrfn(current_step, num_warmup_steps, lr_max, num_cycles=0.50, num_training_steps=5):
    if current_step < num_warmup_steps:
        return lr_max * 0.50 ** (num_warmup_steps - current_step)
    else:
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr_max

if __name__ == '__main__':
    BATCH_SIZE = 32
    # MODEL_NAME = "efficientNet7"
    # MODEL_NAME = "perceptron"
    MODEL_NAME = "conv_layer"

    label_file = pd.read_csv("data/train_labels.csv")
    label_file = label_file[label_file["target"] >= 0]

    label_file_train, label_file_val = train_test_split(label_file, test_size=0.2, random_state=42)

    train_gen = CustomDataGen("train", label_file_train, BATCH_SIZE)
    val_gen = CustomDataGen("train", label_file_val, BATCH_SIZE)

    object = Model()
    model = object.get_model(MODEL_NAME, False)

    model.summary()
    EPOCHS = 2
    LR_SCHEDULE = [lrfn(step, num_warmup_steps=0, lr_max=4e-4, num_cycles=0.50) for step in range(EPOCHS)]
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda step: LR_SCHEDULE[step], verbose=1)
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS,
              callbacks=[lr_callback, WeightDecayCallback(0.01), tf.keras.callbacks.TensorBoard("logs/"+MODEL_NAME)], )
    model.save_weights("model_weights/" + MODEL_NAME + "/" + MODEL_NAME)

    output_train = model.predict(train_gen)
    output_test = model.predict(val_gen)
    output_train = np.where(output_train > 0.5, 1, 0)
    output_test = np.where(output_test > 0.5, 1, 0)
    cm_train = tf.math.confusion_matrix(label_file_train["target"],output_train)
    cm_test = tf.math.confusion_matrix(label_file_val["target"], output_test)
    print(cm_train)
    print(cm_test)