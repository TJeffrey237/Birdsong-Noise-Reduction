import librosa
import matplotlib.pyplot as plt
import numpy as np
import noisereduce as nr
import soundfile as sf
import os

from Spectogram.spectogram import Spectogram
from audio_io import compute_stft

def main():
    print("Start of program")
    file_path = "data/raw/xeno_canto/Song_Sparrow/XC1090712 - Song Sparrow - Melospiza melodia.mp3"
    write_audio_path = "outputs/Song_Sparrow/XC1090712 - Song Sparrow - Melospiza melodia_spectral_gate.wav"
    write_spectrogram_path = "outputs/Song_Sparrow/XC1090712 - Song Sparrow - Melospiza melodia_spectral_gate_spectrogram.png"

    audio, sr = librosa.load(file_path, sr=None)
    reduced_audio = spectral_gate(audio, sr)

    sf.write(write_audio_path, reduced_audio, sr)
    print(f"Saved reduced noise to {write_audio_path}")

    compare_spectrograms(audio, reduced_audio, sr, write_to_path=write_spectrogram_path)

    print("End of program")

def spectral_gate(audio, sr):
    """
    Apply spectral gate logic to a given audio track.

    Wrapper for noisereduce spectral gating, abstracted for consistnecy

    Args:
        audio (np.ndarray): Audio signal to process.
        sr (int): Sample rate of the audio signal.

    Returns:
        np.ndarray: Audio signal with reduced noise.
    """
    return nr.reduce_noise(y=audio, sr=sr, stationary=False)

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

def compare_spectrograms(
    original_audio: np.ndarray,
    processed_audio: np.ndarray,
    sr: int,
    write_to_path: str = None,
    show: bool = True
):
    """
    Compares the spectrograms of an original and processed audio signal.

    Args:
        original_audio (np.ndarray): Original audio waveform.
        processed_audio (np.ndarray): Processed audio waveform.
        sr (int): Sample rate shared by both signals.
        write_to_path (str, optional): Path to save the comparison.
        show (bool): Whether to display the comparison.
    """
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14, 5)
    )

    signals = [
        (original_audio, "Original Audio"),
        (processed_audio, "Processed Audio")
    ]

    for ax, (audio, title) in zip(axes, signals):
        D = librosa.stft(audio)
        S_db = librosa.amplitude_to_db(
            np.abs(D),
            ref=np.max
        )

        img = librosa.display.specshow(
            S_db,
            sr=sr,
            x_axis="time",
            y_axis="hz",
            ax=ax
        )

        ax.set_title(title)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

    fig.tight_layout()

    if write_to_path is not None:
        directory = os.path.dirname(write_to_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        fig.savefig(write_to_path, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)

if __name__ == "__main__":
    main()