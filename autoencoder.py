import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.block1 = ConvBlock(1, 32)
        self.pool1 = nn.MaxPool2d(2)   # 64 -> 32

        self.block2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)   # 32 -> 16

        self.block3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)   # 16 -> 8

        self.block4 = ConvBlock(128, 256)
        self.pool4 = nn.MaxPool2d(2)   # 8 -> 4

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4 * 4 * 256, 256)
        self.fc2 = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = self.pool3(x)

        x = self.block4(x)
        x = self.pool4(x)

        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        z = self.fc2(x)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 4 * 4 * 256)

        self.up1 = DeconvBlock(256, 128)   # 4 -> 8
        self.up2 = DeconvBlock(128, 64)    # 8 -> 16
        self.up3 = DeconvBlock(64, 32)     # 16 -> 32

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(
                32, 16,
                kernel_size=3, stride=2, padding=1, output_padding=1
            ),                              # 32 -> 64
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = x.view(-1, 256, 4, 4)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


def build_autoencoder(latent_dim=128):
    encoder = Encoder(latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim)
    autoencoder = Autoencoder(latent_dim=latent_dim)
    return encoder, decoder, autoencoder


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder, decoder, autoencoder = build_autoencoder(latent_dim=128)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    autoencoder = autoencoder.to(device)

    x = torch.randn(8, 1, 64, 64).to(device)

    z = encoder(x)
    reconstructed = decoder(z)
    output = autoencoder(x)

    print("\nENCODER OUTPUT SHAPE:")
    print(z.shape)                # [8, 128]

    print("\nDECODER OUTPUT SHAPE:")
    print(reconstructed.shape)    # [8, 1, 64, 64]

    print("\nAUTOENCODER OUTPUT SHAPE:")
    print(output.shape)           # [8, 1, 64, 64]

    total_params = sum(p.numel() for p in autoencoder.parameters())
    trainable_params = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")