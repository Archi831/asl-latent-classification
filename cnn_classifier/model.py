# cnn_classifier/model.py
import torch
import torch.nn as nn

NUM_CLASSES = 29

class ConvBlock(nn.Module):
    """Conv2d(3x3) + BN + ReLU, applied twice. Identical to autoencoder.py."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),   nn.MaxPool2d(2),  # → (B,  32, 32, 32)
            ConvBlock(32, 64),  nn.MaxPool2d(2),  # → (B,  64, 16, 16)
            ConvBlock(64, 128), nn.MaxPool2d(2),  # → (B, 128,  8,  8)
            ConvBlock(128, 256),nn.MaxPool2d(2),  # → (B, 256,  4,  4)
            nn.AdaptiveAvgPool2d(1),               # → (B, 256,  1,  1)
            nn.Flatten(),                          # → (B, 256)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),           # raw logits
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def build_classifier(num_classes=NUM_CLASSES, dropout=0.5):
    return CNNClassifier(num_classes=num_classes, dropout=dropout)

if __name__ == "__main__":
    model = build_classifier()
    x = torch.randn(8, 1, 64, 64)
    out = model(x)
    print(f"Output shape: {out.shape}")   # expect (8, 29)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")
