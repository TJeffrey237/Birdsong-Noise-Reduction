import librosa
import numpy as np
import noisereduce as nr
import soundfile as sf

from spectogram.compare_spectograms import compare_spectrograms

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

def spectral_gate(audio: np.ndarray, sr: int) -> np.ndarray:
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

if __name__ == "__main__":
    main()