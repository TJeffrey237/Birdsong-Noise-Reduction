import librosa
import matplotlib.pyplot as plt
import numpy as np

def show_spectrogram(audio, sr, write_to_path=None):
    """
    Utility function to log a single spectrogram from an audio signal.

    Args:
        audio (np.ndarray): Audio signal to display.
        sr (int): Sample rate of the audio signal.
        write_to_path (str, optional): Path to save the spectrogram.
    """
    D = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    librosa.display.specshow(
        S_db,
        sr=sr,
        x_axis="time",
        y_axis="hz"
    )

    plt.colorbar(format="%+2.0f dB")
    if write_to_path is not None:
        plt.savefig(write_to_path)
    plt.show()