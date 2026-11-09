
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os

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