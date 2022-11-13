import tensorflow as tf
import numpy as np
import h5py

class CustomDataGen(tf.keras.utils.Sequence):

    def __init__(self, df, batch_size, shuffle=True):
        self.df = df.copy()
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.n = len(self.df)

    def __len__(self):
        return self.n // self.batch_size

    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __get_input(self, path):
        f = h5py.File(path, "r")

        key0, *_ = f.keys()
        group0 = f[key0]
        key1_H1, key1_L1, key1_frequency_Hz = group0.keys()
        group1_H1 = group0[key1_H1]
        group1_L1 = group0[key1_L1]

        key2_H1_SFTs, key2_H1_timestamps_GPS = group1_H1.keys()
        key2_L1_SFTs, key2_L1_timestamps_GPS = group1_L1.keys()

        # Data
        dataset_H1_SFTs = group1_H1[key2_H1_SFTs]
        # dataset_H1_timestamps_GPS = group1_H1[key2_H1_timestamps_GPS]

        # dataset_L1_SFTs = group1_L1[key2_L1_SFTs]
        # dataset_L1_timestamps_GPS = group1_L1[key2_L1_timestamps_GPS]

        # dataset_frequency_Hz = group0[key1_frequency_Hz]

        return dataset_H1_SFTs[:,:4096]

    def __get_data(self, batches):

        path_batch = "./data/train/" + batches["id"] + ".hdf5"
        label_batch = batches["target"]

        X_batch = np.asarray([self.__get_input(x) for x in path_batch]).astype(np.float64)
        X_batch = np.abs(X_batch)
        X_batch = np.divide(np.subtract(X_batch.T, np.mean(X_batch, axis=(1, 2))), (np.std(X_batch, axis=(1, 2)))).T

        y_batch = np.array(label_batch)

        return X_batch, y_batch

