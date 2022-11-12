import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Model import Model

# Structure des données au format hdf5
# <HDF5 group "/001121a05" (3 members)>
# ----><HDF5 group "/001121a05/H1" (2 members)>
# ---------><HDF5 dataset "SFTs": shape (360, 4612), type "<c8">
# ---------><HDF5 dataset "timestamps_GPS": shape (4612,), type "<i8">
# ----><HDF5 group "/001121a05/L1" (2 members)>
# ---------><HDF5 dataset "SFTs": shape (360, 4653), type "<c8">
# ---------><HDF5 dataset "timestamps_GPS": shape (4653,), type "<i8">
# ----><HDF5 dataset "frequency_Hz": shape (360,), type "<f8">



def read_data_from_hdf5(path):
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
    dataset_H1_timestamps_GPS = group1_H1[key2_H1_timestamps_GPS]

    dataset_L1_SFTs = group1_L1[key2_L1_SFTs]
    dataset_L1_timestamps_GPS = group1_L1[key2_L1_timestamps_GPS]

    dataset_frequency_Hz = group0[key1_frequency_Hz]

    return dataset_H1_SFTs, dataset_H1_timestamps_GPS, dataset_L1_SFTs, dataset_L1_timestamps_GPS, dataset_frequency_Hz

def plot_spectrogram(amplitude):
    plt.title('Spectrogram')
    plt.xlabel('time')
    plt.ylabel('frequency')
    plt.imshow(amplitude)
    plt.colorbar()
    plt.show()

def plot_aplitude_by_freq(amplitude):
    plt.title('Amplitude')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.plot(amplitude[:,0])
    plt.show()

if __name__ == '__main__':
    # filename = "./data/001121a05.hdf5" # label 1
    # filename = "./data/01bcf6533.hdf5"  # label 0

    # files = ["./data/001121a05.hdf5", "./data/01bcf6533.hdf5"]
    # files_id = ["001121a05", "01bcf6533"]


    label_file = pd.read_csv("data/label/train_labels.csv")

    x_train = []
    y_train = []
    for file in os.listdir("./data/train/"):
        x_train.append(np.array(read_data_from_hdf5("./data/train/"+file)[0])[:,:4096])
        y_train.append(label_file[label_file["id"] == os.path.splitext(file)[0]]["target"].values[0])

    x_train = np.abs(x_train)
    x_train = np.divide(np.subtract(x_train.T, np.mean(x_train, axis=(1,2))), (np.std(x_train, axis=(1,2))) ).T
    y_train = np.array(y_train)

    object = Model()
    model_perceptron = object.get_model("perceptron")

    model_perceptron.summary()
    print(x_train.shape)
    print(y_train.shape)

    model_perceptron.fit(x_train, y_train, epochs=10)

    # H1_SFTs, H1_timestamps_GPS, *_, frequency_Hz = read_data_from_hdf5(filename)


    # H1_SFTs = np.abs(np.array(H1_SFTs)) # on récupère le spectre d'amplitude
    # H1_timestamps_GPS = np.array(H1_timestamps_GPS)
    # frequency_Hz = np.array(frequency_Hz)
    #
    # frequency_Hz_normalized = (frequency_Hz - np.min(frequency_Hz))/(np.max(frequency_Hz) - np.min(frequency_Hz))
    # H1_timestamps_GPS_normalized = (H1_timestamps_GPS - np.min(H1_timestamps_GPS))/(np.max(H1_timestamps_GPS) - np.min(H1_timestamps_GPS))
    # # H1_SFTs_normalized = (H1_SFTs - np.mean(H1_SFTs, axis=1)[...,None])/(np.std(H1_SFTs, axis=1))[...,None]
    # H1_SFTs_normalized = (H1_SFTs - np.mean(H1_SFTs)) / (np.std(H1_SFTs))
    # H1_SFTs_normalized = np.mean(H1_SFTs_normalized[:,:4096].reshape(360, 128, 32), axis=2)

    # def func(z):
    #     if z > 2:
    #         return z
    #     else:
    #         return 0

    # H1_SFTs_normalized = np.vectorize(func)(H1_SFTs_normalized)
    # plot_spectrogram(H1_SFTs_normalized)
    # plot_aplitude_by_freq(H1_SFTs_normalized)











