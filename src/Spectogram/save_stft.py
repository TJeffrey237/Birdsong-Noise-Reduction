import numpy as np

def save_stft(file_path, raw_mag, mag_norm, phase):
    np.savez_compressed(
        file_path,
        raw_mag=raw_mag,
        mag_norm=mag_norm,
        phase=phase
    )