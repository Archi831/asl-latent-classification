# cnn_classifier/model.py
import torch
import torch.nn as nn

NUM_CLASSES = 39

class ConvBlock(nn.Module):
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


def _build_head(head: str, in_features: int, num_classes: int, dropout: float) -> nn.Sequential:
    if head == "shallow":
        return nn.Sequential(
            nn.Linear(in_features, num_classes),
        )
    if head == "standard":
        return nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
    if head == "deep":
        return nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
    raise ValueError(f"Unknown head type: {head!r}. Choose 'shallow', 'standard', or 'deep'.")


class CNNClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.5, head="standard"):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),    nn.MaxPool2d(2),   # -> (B,  32, 32, 32)
            ConvBlock(32, 64),   nn.MaxPool2d(2),   # -> (B,  64, 16, 16)
            ConvBlock(64, 128),  nn.MaxPool2d(2),   # -> (B, 128,  8,  8)
            ConvBlock(128, 256), nn.MaxPool2d(2),   # -> (B, 256,  4,  4)
            nn.AdaptiveAvgPool2d(1),                # -> (B, 256,  1,  1)
            nn.Flatten(),                           # -> (B, 256)
        )
        self.classifier = _build_head(head, 256, num_classes, dropout)

    def forward(self, x):
        return self.classifier(self.features(x))


def build_classifier(num_classes=NUM_CLASSES, dropout=0.5, head="standard"):
    return CNNClassifier(num_classes=num_classes, dropout=dropout, head=head)


if __name__ == "__main__":
    for h in ("shallow", "standard", "deep"):
        m = build_classifier(num_classes=39, head=h)
        out = m(torch.randn(4, 1, 64, 64))
        params = sum(p.numel() for p in m.parameters())
        print(f"head={h:8s}  output={out.shape}  params={params:,}")
