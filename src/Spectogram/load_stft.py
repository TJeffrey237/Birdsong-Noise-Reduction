import numpy as np

def load_stft(file_path):

    data = np.load(file_path)

    return (
        data['raw_mag'],
        data['mag_norm'],
        data['phase']
    )