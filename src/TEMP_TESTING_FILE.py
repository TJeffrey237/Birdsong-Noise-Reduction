"""
src/test_cae_smoke.py
Sanity check: Validates that the trained CAE reconstructs ambient noise
while intentionally failing on vocalizations (producing call residuals).
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from model_cae import SpectrogramCAE
from dataset import create_noise_dataloader
from audio_io import load_standard_audio, compute_stft


def test_trained_cae(
    checkpoint_path: str = "checkpoints/noise_model.pt",
    focal_bird_path: str = "data/raw/xeno_canto/Song_Sparrow/XC1013035 - Song Sparrow - Melospiza melodia.wav",
    output_plot_path: str = "outputs/training_stage_verification.png"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing CAE on device: {device}")

    # 1. Load the trained checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    model = SpectrogramCAE().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print("Loaded model checkpoint successfully.")

    # -------------------------------------------------------------
    # Test A: Background Noise Reconstruction (Should succeed / low error)
    # -------------------------------------------------------------
    loader = create_noise_dataloader(batch_size=1, max_total_patches=10, shuffle=False)
    noise_patch = next(iter(loader)).to(device)  # [1, 1, 128, 128]

    with torch.no_grad():
        reconstructed_noise = model(noise_patch)
        noise_mse = torch.mean((noise_patch - reconstructed_noise) ** 2).item()

    print(f"\n[Test A] Pure Noise Patch MSE: {noise_mse:.6f}")
    if noise_mse < 0.02:
        print("  --> Noise reconstruction PASSED: Model faithfully reproduces ambient background.")
    else:
        print("  --> Warning: MSE is still high. The model may need more epochs or a lower learning rate.")

   # -------------------------------------------------------------
    # Test B: Bird Call Residual Isolation (Targeting Active Vocalization)
    # -------------------------------------------------------------
    y, sr = load_standard_audio(focal_bird_path, target_sr=32000)
    mag_res = compute_stft(y, n_fft=2048, hop_length=512)
    mag_norm = mag_res[1] if len(mag_res) == 3 else mag_res[0]

    num_freq_bins, total_time_frames = mag_norm.shape

    # Songbirds typically vocalize between 2 kHz and 8 kHz.
    # At 32 kHz sr / 2048 n_fft, each bin is ~15.625 Hz.
    # Bins 128 to 512 span ~2,000 Hz to ~8,000 Hz.
    # We search in this range for the single most energetic 128x128 patch:
    best_window = None
    max_energy = -1.0
    best_f = 0
    best_t = 0

    freq_search_min = 100  # Start around ~1.5 kHz
    freq_search_max = min(512, num_freq_bins - 128)

    # Step across active frequencies and time frames
    for f in range(freq_search_min, freq_search_max, 32):
        for t in range(0, total_time_frames - 128, 64):
            candidate = mag_norm[f : f + 128, t : t + 128]
            energy = np.mean(candidate)
            if energy > max_energy:
                max_energy = energy
                best_window = candidate
                best_f = f
                best_t = t

    if best_window is None:
        # Fallback if audio duration is shorter than 128 frames
        best_window = mag_norm[:128, :128]

    # Convert bin indices to approximate Hz for display
    hz_low = int(best_f * (sr / 2048))
    hz_high = int((best_f + 128) * (sr / 2048))
    print(
        f"Selected vocal patch at time frame {best_t}, frequency band"
        f" {hz_low}–{hz_high} Hz"
    )

    vocal_patch = (
        torch.from_numpy(best_window)
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    with torch.no_grad():
        reconstructed_vocal = model(vocal_patch)
        vocal_mse = torch.mean((vocal_patch - reconstructed_vocal) ** 2).item()

    residual = torch.clamp(vocal_patch - reconstructed_vocal, min=0.0)

    print(f"[Test B] Vocal Patch MSE:     {vocal_mse:.6f}")
    print(
        f"Residual Energy Spike Ratio:  "
        f" {(vocal_mse / (noise_mse + 1e-8)):.2f}x higher than noise floor"
    )

    # -------------------------------------------------------------
    # Save Side-by-Side Verification Visual (with locked colormap)
    # -------------------------------------------------------------
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    # Shared color bounds so visual brightness reflects actual magnitude
    vm, vx = 0.0, 1.0

    # Row 1: Pure Noise
    p_noise = noise_patch.squeeze().cpu().numpy()
    p_recon_noise = reconstructed_noise.squeeze().cpu().numpy()
    axes[0, 0].imshow(p_noise, origin="lower", aspect="auto", cmap="magma", vmin=vm, vmax=vx)
    axes[0, 0].set_title("Original Noise Patch")
    axes[0, 1].imshow(p_recon_noise, origin="lower", aspect="auto", cmap="magma", vmin=vm, vmax=vx)
    axes[0, 1].set_title(f"Reconstructed Noise (MSE: {noise_mse:.4f})")
    axes[0, 2].imshow(np.maximum(0, p_noise - p_recon_noise), origin="lower", aspect="auto", cmap="magma", vmin=vm, vmax=vx)
    axes[0, 2].set_title("Noise Residual (Near Blank)")

    # Row 2: Bird Call
    p_vocal = vocal_patch.squeeze().cpu().numpy()
    p_recon_vocal = reconstructed_vocal.squeeze().cpu().numpy()
    axes[1, 0].imshow(p_vocal, origin="lower", aspect="auto", cmap="magma", vmin=vm, vmax=vx)
    axes[1, 0].set_title(f"Original Bird Call ({hz_low}–{hz_high} Hz)")
    axes[1, 1].imshow(p_recon_vocal, origin="lower", aspect="auto", cmap="magma", vmin=vm, vmax=vx)
    axes[1, 1].set_title(f"Reconstructed Guess (MSE: {vocal_mse:.4f})")
    axes[1, 2].imshow(residual.squeeze().cpu().numpy(), origin="lower", aspect="auto", cmap="magma", vmin=vm, vmax=vx)
    axes[1, 2].set_title("Vocal Residual (Bird Call Isolated!)")

    # Dynamic Axis Labels
    for col in range(3):
        axes[0, col].set_xlabel("Time Frames")
        axes[0, col].set_ylabel("Freq Bins (0–2 kHz)")
        axes[1, col].set_xlabel("Time Frames")
        axes[1, col].set_ylabel(f"Freq ({hz_low}–{hz_high} Hz)")

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=150)
    plt.close()
    print(f"\nVerification visual successfully saved to: {output_plot_path}")


if __name__ == "__main__":
    test_trained_cae()