import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import random


def plot_spectrogram(amplitude):
    plt.title('Spectrogram')
    plt.xlabel('time')
    plt.ylabel('frequency')
    plt.imshow(amplitude, aspect="auto")
    # plt.imshow(amplitude)
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

def normalize(data):
    data = np.abs(np.array(data)) * 1e22
    data = (data - np.mean(data, axis=(0,1)))/(np.std(data, axis=(0,1)))
    return data
def save_spectrogram():
    tqdm.pandas()
    label_file = pd.read_csv("../../data/train_labels.csv")
    negatif_file = label_file[label_file["target"] == 0]
    positif_file = label_file[label_file["target"] == 1]

    def save_image(row, type):
        dataset_H1_SFTs, _, dataset_L1_SFTs, *_ = read_data_from_hdf5(f'data/train/{str(row["id"])}.hdf5')
        dataset_H1_SFTs = normalize(dataset_H1_SFTs)
        dataset_L1_SFTs = normalize(dataset_L1_SFTs)
        H1_img= Image.fromarray(dataset_H1_SFTs, 'L')
        L1_img= Image.fromarray(dataset_L1_SFTs, 'L')
        H1_img.save(f'data/train_{type}_spectrogram/{row["id"]}_H1.png')
        L1_img.save(f'data/train_{type}_spectrogram/{row["id"]}_L1.png')

    negatif_file.progress_apply(lambda row: save_image(row, "negatif"), axis=1)
    positif_file.progress_apply(lambda row: save_image(row, "positif"), axis=1)

def reduce_noise_and_plot(file_path_image):
    H1_SFTs, _, L1_SFTs, *_ = read_data_from_hdf5(file_path_image)

    H1_SFTs = np.abs(np.array(H1_SFTs)) * 1e22  # on récupère le spectre d'amplitude
    L1_SFTs = np.abs(np.array(L1_SFTs)) * 1e22

    print(H1_SFTs.min())
    print(L1_SFTs.min())
    print(H1_SFTs.max())
    print(L1_SFTs.max())

    amplitude_min = max(H1_SFTs.min(), L1_SFTs.min())
    amplitude_max = min(H1_SFTs.max(), L1_SFTs.max())

    H1_SFTs_clean = H1_SFTs.copy()
    L1_SFTs_clean = L1_SFTs.copy()

    H1_SFTs_normalized = (H1_SFTs - np.mean(H1_SFTs, axis=(0, 1))) / (np.std(H1_SFTs, axis=(0, 1)))
    H1_SFTs_normalized = np.mean(H1_SFTs_normalized[:, :4096].reshape(360, 128, 32), axis=2)

    L1_SFTs_normalized = (L1_SFTs - np.mean(L1_SFTs, axis=(0, 1))) / (np.std(L1_SFTs, axis=(0, 1)))
    L1_SFTs_normalized = np.mean(L1_SFTs_normalized[:, :4096].reshape(360, 128, 32), axis=2)

    def func(z):
        if z > amplitude_max or z < amplitude_min:
            return random.randrange(int(amplitude_min * 10 ** 5), int(amplitude_max * 10 ** 5)) / 10 ** 5
        else:
            return z

    H1_SFTs_clean = np.vectorize(func)(H1_SFTs_clean)
    L1_SFTs_clean = np.vectorize(func)(L1_SFTs_clean)

    H1_SFTs_clean_normalized = (H1_SFTs_clean - np.mean(H1_SFTs_clean, axis=(0, 1))) / (
        np.std(H1_SFTs_clean, axis=(0, 1)))
    H1_SFTs_clean_normalized = np.mean(H1_SFTs_clean_normalized[:, :4096].reshape(360, 128, 32), axis=2)

    L1_SFTs_clean_normalized = (L1_SFTs_clean - np.mean(L1_SFTs_clean, axis=(0, 1))) / (
        np.std(L1_SFTs_clean, axis=(0, 1)))
    L1_SFTs_clean_normalized = np.mean(L1_SFTs_clean_normalized[:, :4096].reshape(360, 128, 32), axis=2)

    fig, axs = plt.subplots(2, 2)
    im1 = axs[0, 0].imshow(H1_SFTs_normalized, aspect="auto")
    axs[0, 0].set_title('H1')
    plt.colorbar(im1, ax=axs[0, 0])
    im2 = axs[0, 1].imshow(L1_SFTs_normalized, aspect="auto")
    axs[0, 1].set_title('L1')
    plt.colorbar(im2, ax=axs[0, 1])
    im3 = axs[1, 0].imshow(H1_SFTs_clean_normalized, aspect="auto")
    axs[1, 0].set_title('H1 clean')
    plt.colorbar(im3, ax=axs[1, 0])
    im4 = axs[1, 1].imshow(L1_SFTs_clean_normalized, aspect="auto")
    axs[1, 1].set_title('L1 clean')
    plt.colorbar(im4, ax=axs[1, 1])

    plt.show()

def transform_image_and_plot(file_path_image, split = 100):
    H1_SFTs, _, L1_SFTs, *_ = read_data_from_hdf5(file_path_image)

    H1_SFTs = np.abs(np.array(H1_SFTs)) * 1e22  # on récupère le spectre d'amplitude
    L1_SFTs = np.abs(np.array(L1_SFTs)) * 1e22

    H1_SFTs_normalized = (H1_SFTs - np.mean(H1_SFTs, axis=(0, 1))) / (np.std(H1_SFTs, axis=(0, 1)))
    H1_SFTs_normalized = np.mean(H1_SFTs_normalized[:, :4096].reshape(360, 128, 32), axis=2)

    L1_SFTs_normalized = (L1_SFTs - np.mean(L1_SFTs, axis=(0, 1))) / (np.std(L1_SFTs, axis=(0, 1)))
    L1_SFTs_normalized = np.mean(L1_SFTs_normalized[:, :4096].reshape(360, 128, 32), axis=2)

    amplitude_min_H1 = H1_SFTs_normalized.min()
    amplitude_max_H1 = H1_SFTs_normalized.max()
    step_H1 = (amplitude_max_H1 - amplitude_min_H1) / split
    amplitude_min_L1 = L1_SFTs_normalized.min()
    amplitude_max_L1 = L1_SFTs_normalized.max()
    step_L1 = (amplitude_max_L1 - amplitude_min_L1) / split

    H1_SFTs_transform = np.zeros((split+1,128))
    L1_SFTs_transform = np.zeros((split+1,128))

    for i in range(128):
        for j in range(360):
            H1_SFTs_transform[int((H1_SFTs_normalized[j,i] - amplitude_min_H1) // step_H1), i] += 1
            L1_SFTs_transform[int((L1_SFTs_normalized[j,i] - amplitude_min_L1) // step_L1), i] += 1


    fig, axs = plt.subplots(2, 2)
    im1 = axs[0, 0].imshow(H1_SFTs_normalized, aspect="auto")
    axs[0, 0].set_title('H1')
    plt.colorbar(im1, ax=axs[0, 0])
    im2 = axs[0, 1].imshow(L1_SFTs_normalized, aspect="auto")
    axs[0, 1].set_title('L1')
    plt.colorbar(im2, ax=axs[0, 1])
    im3 = axs[1, 0].imshow(H1_SFTs_transform, aspect="auto")
    axs[1, 0].set_title('H1 transform')
    plt.colorbar(im3, ax=axs[1, 0])
    im4 = axs[1, 1].imshow(L1_SFTs_transform, aspect="auto")
    axs[1, 1].set_title('L1 transform')
    plt.colorbar(im4, ax=axs[1, 1])

    plt.show()

def reduce_noise_by_similarity(file_path_image):
    def find_closest_noise(row, input):
        filename = f'./data/test_processed/{row["id"]}.npy'
        H1 = np.load(filename)[:,:,0]
        diff = abs(np.sum(abs(H1 - input)))
        row["diff"] = diff
        return row

    id = file_path_image.split(".")[0].split("/")[-1]
    H1_input = np.load(f'./data/test_processed/{id}.npy')[:,:,0]

    label_file = pd.read_csv("../../data/sample_submission.csv")
    label_file = label_file[(label_file.id != id)].reset_index(drop=True) #(label_file.index < 2000) &

    tqdm.pandas()
    label_file = label_file.progress_apply(lambda row: find_closest_noise(row, H1_input), axis=1)
    id_find = np.argmin(label_file["diff"], axis=0)
    row_find = label_file.loc[id_find]
    H1_to_substract = np.load(f'./data/test_processed/{row_find["id"]}.npy')[:,:,0]

    H1_diff = H1_input - H1_to_substract

    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.5)
    ax1 = plt.subplot(gs[0, :2], )
    ax2 = plt.subplot(gs[0, 2:])
    ax3 = plt.subplot(gs[1, 1:3])

    # fig, axs = plt.subplots(2, 2)
    im1 = ax1.imshow(H1_input, aspect="auto")
    ax1.set_title('H1 initial')
    plt.colorbar(im1, ax=ax1)
    im2 = ax2.imshow(H1_to_substract, aspect="auto")
    ax2.set_title('H1 similar')
    plt.colorbar(im2, ax=ax2)
    im3 = ax3.imshow(H1_diff, aspect="auto")
    ax3.set_title('H1 diff')
    plt.colorbar(im3, ax=ax3)

    plt.show()

if __name__ == '__main__':
    # filename = "./data/train/00f36a6ac.hdf5" # label 1
    # filename = "./data/train_processed/00f36a6ac.npy"  # label 1
    # filename = "./data/train/02c8f43f3.hdf5"
    # filename = "./data/train/01bcf6533.hdf5"  # label 0
    filename = "../../data/test/3cc6680fb.hdf5"
    # filename = "data/test/2083f23b4.hdf5"
    # filename = "data/test/00222d97b.hdf5"

    dataset_H1_SFTs, dataset_H1_timestamps_GPS, dataset_L1_SFTs, dataset_L1_timestamps_GPS, dataset_frequency_Hz_ = read_data_from_hdf5(filename)
    plot_spectrogram(np.abs(dataset_H1_SFTs)*1e22)
