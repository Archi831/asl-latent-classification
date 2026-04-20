import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.up(x))


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()

        self.block1 = ConvBlock(1,   32)
        self.pool1  = nn.MaxPool2d(2)

        self.block2 = ConvBlock(32,  64)
        self.pool2  = nn.MaxPool2d(2)

        self.block3 = ConvBlock(64,  128)
        self.pool3  = nn.MaxPool2d(2)

        self.block4 = ConvBlock(128, 256)
        self.pool4  = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(4 * 4 * 256, 512)
        self.drop    = nn.Dropout(0.3)
        self.fc2     = nn.Linear(512, latent_dim)

        # BatchNorm on z keeps scale consistent across training and inference
        self.bn_z = nn.BatchNorm1d(latent_dim)

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        z = self.fc2(x)
        z = self.bn_z(z)

        # L2-normalise: projects every embedding onto the unit hypersphere.
        # Stops a few dimensions from dominating and makes class clusters
        # tighter — exactly what the downstream MLP benefits from.
        z = F.normalize(z, p=2, dim=1)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()

        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 4 * 4 * 256)

        self.up1 = UpBlock(256, 128)
        self.up2 = UpBlock(128, 64)
        self.up3 = UpBlock(64,  32)
        self.up4 = UpBlock(32,  16)

        self.head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 256, 4, 4)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.head(x)


class ClassifierHead(nn.Module):
    """
    Lightweight head attached to z *during autoencoder training only*.
    Forces the encoder to produce class-discriminative embeddings.
    Discarded after training — your downstream MLP replaces it entirely.
    """
    def __init__(self, latent_dim: int, num_classes: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, z):
        return self.net(z)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim: int = 128, num_classes: int = 39):
        super().__init__()
        self.encoder    = Encoder(latent_dim=latent_dim)
        self.decoder    = Decoder(latent_dim=latent_dim)
        # Included in the model so the optimizer trains it automatically
        self.classifier = ClassifierHead(latent_dim, num_classes)

    def forward(self, x):
        z      = self.encoder(x)
        x_hat  = self.decoder(z)
        logits = self.classifier(z)
        return x_hat, logits       # training loop uses both

    def encode(self, x):
        return self.encoder(x)     # save_latents.py uses this — no change needed


def build_autoencoder(latent_dim: int = 128, num_classes: int = 39):
    encoder     = Encoder(latent_dim=latent_dim)
    decoder     = Decoder(latent_dim=latent_dim)
    autoencoder = Autoencoder(latent_dim=latent_dim, num_classes=num_classes)
    return encoder, decoder, autoencoder


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder, decoder, autoencoder = build_autoencoder(latent_dim=128, num_classes=39)
    autoencoder = autoencoder.to(device)

    x = torch.randn(8, 1, 64, 64).to(device)
    x_hat, logits = autoencoder(x)
    z = autoencoder.encode(x)

    print("Latent shape   :", z.shape)        # (8, 128)
    print("L2 norms (≈1)  :", z.norm(dim=1))  # should all be ~1.0
    print("Recon shape    :", x_hat.shape)    # (8, 1, 64, 64)
    print("Logits shape   :", logits.shape)   # (8, 29)

    total     = sum(p.numel() for p in autoencoder.parameters())
    trainable = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
    print(f"Total params   : {total:,}")
    print(f"Trainable      : {trainable:,}")