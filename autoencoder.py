import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # 64 -> 32
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 16 -> 8
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 8 -> 4
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4 * 4 * 256, 128)
        self.fc2 = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        latent_vector = self.fc2(x)
        return latent_vector


class Decoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 4 * 4 * 256)

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 4 -> 8
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),   # 8 -> 16
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),    # 16 -> 32
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),      # 32 -> 64
            nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.relu(self.fc(x))
        x = x.view(-1, 256, 4, 4)
        x = self.deconv_layers(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


def build_autoencoder(latent_dim=64):
    encoder = Encoder(latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim)
    autoencoder = Autoencoder(latent_dim=latent_dim)
    return encoder, decoder, autoencoder


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder, decoder, autoencoder = build_autoencoder(latent_dim=64)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    autoencoder = autoencoder.to(device)

    x = torch.randn(8, 1, 64, 64).to(device)

    latent = encoder(x)
    reconstructed = decoder(latent)
    output = autoencoder(x)

    print("\nENCODER OUTPUT SHAPE:")
    print(latent.shape)          # expected: [8, 64]

    print("\nDECODER OUTPUT SHAPE:")
    print(reconstructed.shape)   # expected: [8, 1, 64, 64]

    print("\nAUTOENCODER OUTPUT SHAPE:")
    print(output.shape)          # expected: [8, 1, 64, 64])

    total_params = sum(p.numel() for p in autoencoder.parameters())
    trainable_params = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")