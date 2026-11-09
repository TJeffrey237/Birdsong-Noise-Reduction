import os
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from model_cae import SpectrogramCAE
from dataset import create_noise_dataloader

def train(
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 1e-3,
    soundscape_dir: str = "data/raw/birdclef_soundscapes",
    checkpoint_dir: str = "checkpoints",
    model_name: str = "noise_model.pt"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"Device Name: {torch.cuda.get_device_name(0)}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, model_name)

    print("Loading the noise training patches...")
    loader = create_noise_dataloader(
        soundscape_dir = soundscape_dir,
        patch_size = 128,
        batch_size = batch_size,
        shuffle = True
    )

    if len(loader) == 0:
        raise RuntimeError("No batches found.")

    model = SpectrogramCAE().to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr = lr)

    print(f"Starting training for {epochs} epochs.")
    print(f"Batch size: {batch_size}, Learning Rate: {lr}")

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        prog_bar = tqdm(loader, desc = f"Epoch [{epoch:02d}/{epochs:02d}]", leave = False)
        for batch in prog_bar:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstruction  = model(batch)
            loss = criterion(reconstruction , batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            prog_bar.set_postfix({"batch_loss": f"{loss.item():.5f}"})

        avg_loss = running_loss / len(loader)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Average MSE Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Chkpt saved at epoch {epoch:02d} with loss {best_loss:.6f} to {checkpoint_path}")

    print(f"Training completed. Best loss: {best_loss:.6f}")
    print(f"Best model weights saved to: {checkpoint_path}")

if __name__ == "__main__":
    train(
        epochs = 30,
        batch_size = 32,
        lr = 1e-3,
        soundscape_dir = "data/raw/birdclef_soundscapes",
    )