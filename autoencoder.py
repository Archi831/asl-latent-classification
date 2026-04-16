import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic building block with two convolutional layers, batch norm, and ReLU
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), # look for important visual patterns
            nn.BatchNorm2d(out_channels), # organize the detected signals so they are easier to learn from
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) # makes the feature map larger
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class Encoder(nn.Module):

    def __init__(self, latent_dim: int = 128):
        super().__init__()

        self.block1 = ConvBlock(1,   32)   # → (B, 32,  64, 64)
        self.pool1  = nn.MaxPool2d(2)      # → (B, 32,  32, 32) # makes the image representation smaller

        self.block2 = ConvBlock(32,  64)   # → (B, 64,  32, 32)
        self.pool2  = nn.MaxPool2d(2)      # → (B, 64,  16, 16)

        self.block3 = ConvBlock(64,  128)  # → (B, 128, 16, 16)
        self.pool3  = nn.MaxPool2d(2)      # → (B, 128,  8,  8)

        self.block4 = ConvBlock(128, 256)  # → (B, 256,  8,  8)
        self.pool4  = nn.MaxPool2d(2)      # → (B, 256,  4,  4)

        self.flatten = nn.Flatten() # converts the 3D feature map into one long vector for each image
        self.fc1 = nn.Linear(4 * 4 * 256, 512) # compresses the extracted features into a smaller representation
        self.drop = nn.Dropout(0.3) # randomly turns off 30% of neurons during training
        self.fc2 = nn.Linear(512, latent_dim)

    def forward(self, x):
        s1 = self.block1(x)          # skip 1  (B, 32,  64, 64)
        x  = self.pool1(s1)

        s2 = self.block2(x)          # skip 2  (B, 64,  32, 32)
        x  = self.pool2(s2)

        s3 = self.block3(x)          # skip 3  (B, 128, 16, 16)
        x  = self.pool3(s3)

        s4 = self.block4(x)          # skip 4  (B, 256,  8,  8)
        x  = self.pool4(s4)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        z = self.fc2(x)

        self._skips = (s1, s2, s3, s4)
        return z

class Decoder(nn.Module):

    def __init__(self, latent_dim: int = 128):
        super().__init__()

        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 4 * 4 * 256)

        # skip_channels=256 for up1, 128 for up2, 64 for up3, 32 for up4
        self.up1 = UpBlock(256, 256, 128)   # 4  → 8,   concat s4
        self.up2 = UpBlock(128, 128, 64)    # 8  → 16,  concat s3
        self.up3 = UpBlock(64,  64,  32)    # 16 → 32,  concat s2
        self.up4 = UpBlock(32,  32,  16)    # 32 → 64,  concat s1

        self.head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, z, skips=None):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 256, 4, 4)

        s1, s2, s3, s4 = skips if skips is not None else (None, None, None, None)

        x = self.up1(x, s4)
        x = self.up2(x, s3)
        x = self.up3(x, s2)
        x = self.up4(x, s1)

        return self.head(x)

class Autoencoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

    def forward(self, x):
        z      = self.encoder(x)
        x_hat  = self.decoder(z, skips=self.encoder._skips)
        return x_hat

    def encode(self, x):
        z = self.encoder(x)
        return z

def build_autoencoder(latent_dim: int = 128):
    encoder     = Encoder(latent_dim=latent_dim)
    decoder     = Decoder(latent_dim=latent_dim)
    autoencoder = Autoencoder(latent_dim=latent_dim)
    return encoder, decoder, autoencoder

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder, decoder, autoencoder = build_autoencoder(latent_dim=128)
    autoencoder = autoencoder.to(device)

    x = torch.randn(8, 1, 64, 64).to(device)
    z     = autoencoder.encoder(x)
    x_hat = autoencoder(x)

    print("Latent shape      :", z.shape)     # (8, 64)
    print("Reconstruction    :", x_hat.shape) # (8, 1, 64, 64)

    total     = sum(p.numel() for p in autoencoder.parameters())
    trainable = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
    print(f"Total parameters  : {total:,}")
    print(f"Trainable params  : {trainable:,}")