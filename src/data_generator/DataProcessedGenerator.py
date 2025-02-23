import tensorflow as tf
import numpy as np

class DataProcessedGenerator(tf.keras.utils.Sequence):

    def __init__(self, data_type, df, batch_size, L1_H1_mean=False):
        self.data_type = data_type
        self.df = df.copy()
        self.batch_size = batch_size
        self.n = len(self.df)
        self.L1_H1_mean = L1_H1_mean

    def __len__(self):
        if self.n % self.batch_size == 0:
            return self.n // self.batch_size
        else:
            return (self.n // self.batch_size)+1

    def __getitem__(self, index):
        if (self.n // self.batch_size) + 1 == index:
            batches = self.df[(index-1) * self.batch_size:-1]
        else:
            batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __get_data(self, batches):

        path_batch = "data/"+self.data_type+"/"+batches["id"]+".npy"
        label_batch = batches["target"]

        X_batch = np.asarray([np.load(x) for x in path_batch])
        y_batch = np.array(label_batch)

        if self.L1_H1_mean:
            tmp = np.zeros((X_batch.shape[0]*2, 360, 128))
            tmp[0:X_batch.shape[0],:,:] = X_batch[:,:,:,0]
            tmp[X_batch.shape[0]:X_batch.shape[0]*2+1,:,:] = X_batch[:,:,:,1]

            X_batch = tmp
            y_batch = np.concatenate((y_batch.copy(), y_batch.copy()), axis=0)

        return X_batch, y_batch


# Structure des données au format hdf5
# <HDF5 group "/001121a05" (3 members)>
# ----><HDF5 group "/001121a05/H1" (2 members)>
# ---------><HDF5 dataset "SFTs": shape (360, 4612), type "<c8">
# ---------><HDF5 dataset "timestamps_GPS": shape (4612,), type "<i8">
# ----><HDF5 group "/001121a05/L1" (2 members)>
# ---------><HDF5 dataset "SFTs": shape (360, 4653), type "<c8">
# ---------><HDF5 dataset "timestamps_GPS": shape (4653,), type "<i8">
# ----><HDF5 dataset "frequency_Hz": shape (360,), type "<f8">

