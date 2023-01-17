import h5py
import numpy as np
from tqdm import tqdm
import os
import random

from data_visualisation import read_data_from_hdf5


def reduce_noise(H1_L1):

    amplitude_min = max(H1_L1[:, :, 0].min(), H1_L1[:, :, 1].min())
    amplitude_max = min(H1_L1[:, :, 0].max(), H1_L1[:, :, 1].max())

    H1_L1_clean = H1_L1.copy()

    def func(z):
        if z > amplitude_max or z < amplitude_min:
            return random.randrange(int(amplitude_min * 10 ** 5), int(amplitude_max * 10 ** 5)) / 10 ** 5
        else:
            return z

    H1_L1_clean = np.vectorize(func)(H1_L1_clean)

    return H1_L1_clean

def save_data(data_path, data_processed_path, reduce_noise_param=False):
    for file in tqdm(os.listdir(data_path)):
        H1, _, L1, *_ = read_data_from_hdf5(os.path.join(data_path,file))
        H1_L1 = np.zeros((360, 4096, 2), dtype=complex)
        H1_L1[:, :, 0] = np.array(H1)[:, :4096]
        H1_L1[:, :, 1] = np.array(L1)[:, :4096]

        H1_L1 = np.abs(np.array(H1_L1)) * 1e22
        print(H1_L1.mean())
        if reduce_noise_param:
            H1_L1 = reduce_noise(H1_L1)
        print(H1_L1.mean())
        H1_L1 = (H1_L1 - np.mean(H1_L1, axis=(0, 1, 2)))/(np.std(H1_L1, axis=(0, 1, 2)))
        H1_L1 = np.mean(H1_L1.reshape(360, 128, 32, 2), axis=2)

        np.save(os.path.join(data_processed_path,file.split(".")[0]+".npy"), H1_L1)

if __name__ == '__main__':
    data_train_path = "data/train/"
    data_test_path = "data/test/"

    data_train_processed_path = "data/train_processed"
    data_test_processed_path = "data/test_processed"
    data_test_clean_path = "data/test_clean"

    # save_data(data_train_path, data_train_processed_path)
    save_data(data_test_path, data_test_clean_path, reduce_noise_param=True)




