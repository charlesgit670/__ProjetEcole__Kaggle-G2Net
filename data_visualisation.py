import matplotlib.pyplot as plt
import h5py
import numpy as np


def plot_spectrogram(amplitude):
    plt.title('Spectrogram')
    plt.xlabel('time')
    plt.ylabel('frequency')
    plt.imshow(amplitude)
    plt.colorbar()
    plt.show()

def plot_aplitude_by_freq(amplitude):
    plt.title('Amplitude')
    plt.xlabel('frequency')
    plt.ylabel('amplitude')
    plt.plot(amplitude[:,0])
    plt.show()


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

if __name__ == '__main__':
    filename = "./data/train/001121a05.hdf5" # label 1
    # filename = "./data/01bcf6533.hdf5"  # label 0

    H1_SFTs, H1_timestamps_GPS, *_, frequency_Hz = read_data_from_hdf5(filename)


    H1_SFTs = np.abs(np.array(H1_SFTs))*1e22 # on récupère le spectre d'amplitude
    # H1_timestamps_GPS = np.array(H1_timestamps_GPS)
    # frequency_Hz = np.array(frequency_Hz)

    # frequency_Hz_normalized = (frequency_Hz - np.min(frequency_Hz))/(np.max(frequency_Hz) - np.min(frequency_Hz))
    # H1_timestamps_GPS_normalized = (H1_timestamps_GPS - np.min(H1_timestamps_GPS))/(np.max(H1_timestamps_GPS) - np.min(H1_timestamps_GPS))
    H1_SFTs_normalized = (H1_SFTs - np.mean(H1_SFTs, axis=1)[...,None])/(np.std(H1_SFTs, axis=1))[...,None]
    # H1_SFTs_normalized = np.mean(H1_SFTs_normalized[:,:4096].reshape(360, 128, 32), axis=2)

    # def func(z):
    #     if z > 2:
    #         return z
    #     else:
    #         return 0

    # H1_SFTs_normalized = np.vectorize(func)(H1_SFTs_normalized)
    plot_spectrogram(H1_SFTs_normalized)
    plot_aplitude_by_freq(H1_SFTs_normalized)