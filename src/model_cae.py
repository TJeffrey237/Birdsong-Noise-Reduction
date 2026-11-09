import torch
import torch.nn as nn

class SpectrogramCAE(nn.Module):
    def __init__(self):
        super().__init__()

        # this takes a 2D spectrogram input of shape [1, 128, 128]
        # compresses it down to a shape of [64, 16, 16] and then back to the starting shape
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True),

            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True),

            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True),

            nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True),

            nn.ConvTranspose2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True),

            nn.ConvTranspose2d(in_channels = 32, out_channels = 16, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True),

            nn.ConvTranspose2d(in_channels = 16, out_channels = 1, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.Sigmoid()
        )

    # forward pass through the autoencoder
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

# For tesing only NOT for use in production
if __name__ == "__main__":
    model = SpectrogramCAE()
    dummy_input = torch.randn(2, 1, 128, 128)
    output = model(dummy_input)

    assert output.shape == dummy_input.shape, (
        f"Mismatched output shape: expected {dummy_input.shape}, got {output.shape}"
    )

    print("model_cae.py self-test passed.")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")