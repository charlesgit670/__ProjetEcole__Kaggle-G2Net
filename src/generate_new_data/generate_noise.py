import os
import numpy as np
import math
import random
import multiprocessing

import pyfstat


def get_signal_config():
    t_start = 1238166018

    # These parameters describe background noise and data format
    writer_kwargs = {
        "outdir": "folder_tmp_"+multiprocessing.current_process().name,
        'sqrtSX': 1e-23,  # Single-sided Amplitude Spectral Density of the noise
        'Tsft': 1800,  # Fourier transform time duration
        'F0': np.random.uniform(50, 500),
        'Band': 0.2,
        "SFTWindowType": "tukey",  # Window function to compute short Fourier transforms
        "SFTWindowBeta": 0.01,  # Parameter associated to the window function
        'timestamps': {
            'H1': t_start + 1800 * np.arange(1,4097),
            'L1': t_start + 1800 * np.arange(1,4097),
        }
    }

    return writer_kwargs

def generate_cw(id):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    writer_kwargs = get_signal_config()

    writer = pyfstat.BinaryModulatedWriter(**writer_kwargs)

    writer.make_data() # generate file used by SignalToNoiseRatio.from_sfts

    # Data can be read as a numpy array using PyFstat
    frequency, timestamps, amplitudes = pyfstat.utils.get_sft_as_arrays(
        writer.sftfilepath
    )

    # Cast to complex 128
    for detector, signal in amplitudes.items():
        amplitudes[detector] = signal.astype(np.complex128)

    signals = np.zeros((amplitudes['H1'].shape[0], 4096, 2), dtype=complex)
    signals[:, :, 0] = np.array(amplitudes['H1'])
    signals[:, :, 1] = np.array(amplitudes['L1'])

    signals = np.abs(signals) * 1e22
    signals = (signals - np.mean(signals, axis=(0, 1, 2))) / (np.std(signals, axis=(0, 1, 2)))
    signals = np.mean(signals.reshape(-1, 128, 32, 2), axis=2)

    offset_y = np.random.randint(0, signals.shape[0] - 360)
    signals = signals[offset_y:offset_y + 360, :,:]

    np.save(f"data_noise/signal_noise_{id}.npy", signals)


if __name__ == '__main__':
    # Generate samples in parallel
    num_generated_data = 32
    NUMBER_OF_PROCESSES = multiprocessing.cpu_count()
    print(NUMBER_OF_PROCESSES)

    pool = multiprocessing.Pool(processes=NUMBER_OF_PROCESSES)
    try:
        map_jobs = [(i,) for i in range(0, num_generated_data)]
        result = pool.starmap(generate_cw, map_jobs)
    finally:
        pool.close()
        pool.join()