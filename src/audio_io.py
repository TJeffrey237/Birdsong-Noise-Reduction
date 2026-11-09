"""This file handles audio ingestion, STFT/ISTFT transforms, and phase caching."""
import numpy as np
import librosa
import soundfile as sf

# Loads and resample the audio file to a target sample rate (32kHz) and converts to mono.
def load_standard_audio(file_path: str, target_sr: int = 32000):
    y, sr = librosa.load(file_path, sr = target_sr, mono=True)
    return y, sr

# Does STFT and returns normalized magnitude and phase.
def compute_stft(y: np.ndarray, n_fft: int = 2048, hop_length: int = 512):
    D = librosa.stft(y, n_fft = n_fft, hop_length = hop_length)
    raw_mag = np.abs(D)
    phase = np.angle(D)

    db = librosa.amplitude_to_db(raw_mag, ref = np.max)

    # Normalize magnitude to [0, 1]
    db_min = db.min()
    db_max = db.max()
    mag_norm = (db - db_min) / (db_max - db_min + 1e-8)

    return raw_mag, mag_norm, phase

# Conducts ISTFT given the normalized mag and phase, retuns reconstructed audio.
def synthesize_istft(mag: np.ndarray, phase: np.ndarray, hop_length: int = 512) -> np.ndarray:
    D_reconstructed = mag * np.exp(1j * phase)
    return librosa.istft(D_reconstructed, hop_length = hop_length)

# For testing only NOT for use in production.
if __name__ == "__main__":
    import os
    from glob import glob

    test_audio_path = "data/raw/xeno_canto/Song_Sparrow/XC1190613 - Song Sparrow - Melospiza melodia fallax.wav"

    if not os.path.exists(test_audio_path):
        available_files = (
            glob("data/raw/xeno_canto/*/*.wav") +
            glob("data/raw/xeno_canto/*/*.mp3") +
            glob("data/raw/birdclef_soundscapes/*.*")
        )
        if available_files:
            test_audio_path = available_files[0]
            print(f"Using found audio file: {test_audio_path}")
        else:
            raise FileNotFoundError("No audio files found in data/raw/ to test with.")

    y, sr = load_standard_audio(test_audio_path, target_sr=32000)
    raw_mag, mag_norm, phase = compute_stft(y, n_fft=2048, hop_length=512)
    y_reconstructed = synthesize_istft(raw_mag, phase, hop_length=512)

    min_len = min(len(y), len(y_reconstructed))
    mse = np.mean((y[:min_len] - y_reconstructed[:min_len]) ** 2)
    print(f"Reconstruction Mean Squared Error (MSE): {mse:.2e}")

    os.makedirs("outputs", exist_ok=True)
    out_file = "outputs/audio_io_test_reconstruction.wav"
    sf.write(out_file, y_reconstructed, sr)
    print(f"Saved clean round-trip audio to: {out_file}")