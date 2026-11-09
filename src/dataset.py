import os
from glob import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from audio_io import load_standard_audio, compute_stft

# this is a dataset of 2D magnitude spectrogram patches containing
# ambient noise
class AmbientNoisePatchDataset(Dataset):
    def __init__(self, patches: np.ndarray):
        self.patches = torch.from_numpy(patches).float()

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.patches[idx]

def extract_noise_patches(
    file_path: str,
    patch_size: int = 128,
    target_sr: int = 32000,
    n_fft: int = 2048,
    hop_length: int = 512,
    max_patches: int = 40,
) -> list[np.ndarray]:
    y, sr = load_standard_audio(file_path, target_sr = target_sr)
    stft_res = compute_stft(y, n_fft = n_fft, hop_length = hop_length)

    if len(stft_res) == 3:
        mag_norm = stft_res[1]
    else:
        mag_norm = stft_res[0]

    num_frequency_bins = mag_norm.shape[0]
    total_time_frames = mag_norm.shape[1]
    patches = []

    if num_frequency_bins < patch_size or total_time_frames < patch_size:
        return patches

    for t in range(0, total_time_frames - patch_size + 1, patch_size):
        patch = mag_norm[:patch_size, t:t + patch_size]
        patch = np.expand_dims(patch, axis = 0)
        patches.append(patch)

        if len(patches) >= max_patches:
            break
            
    return patches

# creates a dataloader that yields the batches of ambient noise patches that
# were extracted from the birdCLEF soundscape files
def create_noise_dataloader(
    soundscape_dir: str = "data/raw/birdclef_soundscapes",
    # labels_csv_path: str = "data/raw/train_soundscape_labels.csv",
    patch_size: int = 128,
    batch_size: int = 32,
    max_total_patches: int = 3000,
    shuffle: bool = True,
) -> DataLoader:
    audio_files = (
        glob(os.path.join(soundscape_dir, "*.wav")) +
        glob(os.path.join(soundscape_dir, "*.mp3")) +
        glob(os.path.join(soundscape_dir, "*.ogg*"))
    )

    # uses synthetic noise patches if the soundscape files cannot be found
    if not audio_files:
        print(f"No soundscape files found in {soundscape_dir}.")
        print(f"Using a synthetic noise dataset instead...")

        synthetic_patches = np.random.uniform(0.0, 0.2, (64, 1, patch_size, patch_size)).astype(np.float32)
        dataset = AmbientNoisePatchDataset(synthetic_patches)
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
        return dataloader

    print(f"Extracting noise patches from {len(audio_files)} soundscape files...")
    all_patches = []

    for f in audio_files:
        patches = extract_noise_patches(f, patch_size = patch_size)
        all_patches.extend(patches)

        if len(all_patches) >= max_total_patches:
            all_patches = all_patches[:max_total_patches]
            break

    patches_array = np.array(all_patches, dtype = np.float32)
    print(f"Successfully extracted {len(patches_array)} patches of shape {patches_array.shape[1:]}.")

    dataset = AmbientNoisePatchDataset(patches_array)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)

    return dataloader

# for testing only NOT for use in production
if __name__ == "__main__":
    loader = create_noise_dataloader(batch_size = 8)
    sample_batch = next(iter(loader))

    print("\nsrc/dataset.py test passed.")
    print(f"Batch shape: {sample_batch.shape} (Expected: [8, 1, 128, 128])")
    print(f"Value range: [{sample_batch.min():.3f}, {sample_batch.max():.3f}] (Expected within [0.0, 1.0])")