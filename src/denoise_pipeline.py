"""
THIS IS YOUR MAIN FILE FOR RUNNING THE MODEL
NEEDS: 
    AN INPUT AUDIO FILE (.wav, .mp3) PATH
    TRAINING CHECKPOINT (.pt) INSIDE CHECKPOINTS DIRECTORY
OUTPUTS:
    DENOISED .wav FILE IN OUTPUTS DIR
    SPECTROGRAM VISUALS IN OUTPUTS DR    
"""

import os
import torch
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from audio_io import load_standard_audio, compute_stft, synthesize_istft
from model_cae import SpectrogramCAE

def compute_soft_mask(residual: np.ndarray, gain: float = 12.0, bias: float = 0.12) -> np.ndarray:
    mask = 1.0 / (1.0 + np.exp(-gain * (residual - bias)))
    return np.clip(mask, 0.0, 1.0)

def denoise_audio(
    input_audio_path: str,
    output_audio_path: str,
    checkpoint_path: str = "checkpoints/noise_model.pt",
    gain: float = 14.0,
    bias: float = 0.10,
    patch_size: int = 128,
    save_plot: bool = True,
    plot_path: str = "outputs/denoising_plot.png"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running denoising on device: {device}")

    # load the audio
    y, sr = load_standard_audio(input_audio_path, target_sr = 32000)
    raw_mag, mag_norm, phase = compute_stft(y, n_fft = 2048, hop_length = 512)

    num_bins = mag_norm.shape[0]
    num_frames = mag_norm.shape[1]

    # load the trained CAE
    model = SpectrogramCAE().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location = device))
    model.eval()

    # estimate the full bg noise floor on the normalized spectrogram
    reconstructed_noise = np.zeros_like(mag_norm)
    overlap_counts = np.zeros_like(mag_norm)
    step = patch_size // 2

    with torch.no_grad():
        for f in range (0, num_bins - patch_size + 1, step):
            for t in range(0, num_frames - patch_size + 1, step):
                patch = mag_norm[f:f + patch_size, t:t + patch_size]
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)

                recon_patch = model(patch_tensor).squeeze().cpu().numpy()

                reconstructed_noise[f:f + patch_size, t:t + patch_size] += recon_patch
                overlap_counts[f:f + patch_size, t:t + patch_size] += 1.0

    # normalize the overlapping regions
    nonzero_mask = overlap_counts > 0
    reconstructed_noise[nonzero_mask] /= overlap_counts[nonzero_mask]
    reconstructed_noise[~nonzero_mask] = np.median(mag_norm)

    # get positive residual & soft transmission mask
    residual = np.maximum(0.0, mag_norm - reconstructed_noise)
    soft_mask = compute_soft_mask(residual, gain = gain, bias = bias)

    # apply mask to the orignial magnitude
    clean_mag_linear = raw_mag * soft_mask
    y_clean = synthesize_istft(clean_mag_linear, phase, hop_length = 512)

    clean_mag_db = librosa.amplitude_to_db(clean_mag_linear, ref = np.max)
    clean_mag_vis = (clean_mag_db - clean_mag_db.min()) / (clean_mag_db.max() - clean_mag_db.min() + 1e-8)

    # save the denoised .wav file
    os.makedirs(os.path.dirname(output_audio_path) or ".", exist_ok = True)
    sf.write(output_audio_path, y_clean, sr)
    print(f"Saved output audio file to: {output_audio_path}")

    # save and export the plot
    if save_plot:
        os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
        fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True, sharey=True)

        axs[0].imshow(mag_norm, origin="lower", aspect="auto", cmap="magma")
        axs[0].set_title("1. Raw Noisy Spectrogram")

        axs[1].imshow(soft_mask, origin="lower", aspect="auto", cmap="viridis")
        axs[1].set_title("2. Computed 2D Soft Transmission Mask [0.0, 1.0]")

        axs[2].imshow(clean_mag_vis, origin="lower", aspect="auto", cmap="magma")
        axs[2].set_title("3. Denoised Magnitude Spectrogram (Cleaned)")

        for ax in axs:
            ax.set_ylabel("Frequency Bins")
        axs[2].set_xlabel("Time Frames")

        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Comparison spectrogram plot saved to: {plot_path}")

if __name__ == "__main__":
    # set this for which audio file you want to use
    test_input = "data/raw/xeno_canto/American_Robin/XC860362 - American Robin - Turdus migratorius.mp3"
    test_output = "outputs/denoised_test_bird.wav"

    if os.path.exists(test_input):
        denoise_audio(test_input, test_output)
    else:
        print(f"Error: Invalid input file path.")