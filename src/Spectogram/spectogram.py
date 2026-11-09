import numpy as np
import librosa
import matplotlib.pyplot as plt

def Spectogram(mag_norm: np.ndarray, sr: int = 32000, hop_length: int = 512):
    """
    Display a normalized magnitude spectrogram.

    Parameters:
        mag_norm: Normalized magnitude spectrogram from compute_stft().
        sr: Audio sampling rate.
        hop_length: Number of samples between successive STFT frames.
    """

    plt.figure(figsize=(12, 5))

    librosa.display.specshow(
        mag_norm,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='linear',
        cmap='magma'
    )

    plt.colorbar(label='Normalized Magnitude')
    plt.title('Bird Song Spectrogram')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()
