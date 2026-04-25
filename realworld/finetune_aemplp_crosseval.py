"""
AE+MLP cross-dataset evaluation: zero-shot, MLP-retrain, and encoder+MLP fine-tune.

Mirrors finetune_crosseval.py but for the AE+MLP pipeline. Produces per-class
accuracy CSVs compatible with the existing CNN cross-eval results.

Usage:
    python finetune_aemplp_crosseval.py --mode zeroshot
    python finetune_aemplp_crosseval.py --mode mlp_only
    python finetune_aemplp_crosseval.py --mode full
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import kagglehub
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "autoencoder"))
from autoencoder import Encoder, ClassifierHead, Autoencoder  # noqa: E402

LATENT_DIM  = 128
NUM_CLASSES = 39
BATCH_SIZE  = 64
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AE_PATH      = HERE / "realworld" / "models" / "final_autoencoder.pth"
MLP_PATH     = HERE / "realworld" / "models" / "asl_mlp_model.pth"
AE_BEST_PATH = HERE / "autoencoder" / "models" / "best_autoencoder_ae_asl39_ld128.pth"

OUT_DIR_ZEROSHOT = HERE / "realworld" / "results" / "zeroshot"
OUT_DIR_FINETUNE = HERE / "realworld" / "results" / "finetune"

CLASS_NAMES = [
    "0","1","2","3","4","5","6","7","8","9",
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "nothing","space","unknown",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DATASETS = {
    "ayuraj":     "ayuraj/asl-dataset",
    "danrasband": "danrasband/asl-alphabet-test",
}

FINETUNE_MAX_PER_CLASS: dict[str, int] = {}

eval_tf = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

train_tf = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=8),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
])


# ── dataset ───────────────────────────────────────────────────────────────────

class ImageSamples(Dataset):
    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), label


def has_class_dirs(path: Path) -> bool:
    try:
        return any(d.is_dir() and d.name.upper() in CLASS_TO_IDX for d in path.iterdir())
    except Exception:
        return False


def find_dataset_root(base: Path) -> Path:
    queue = [base]
    for _ in range(4):
        next_q = []
        for p in queue:
            if has_class_dirs(p):
                return p
            try:
                next_q.extend(d for d in sorted(p.iterdir()) if d.is_dir())
            except Exception:
                pass
        queue = next_q
    return base


def collect_samples(root: Path, max_per_class: int | None = None) -> list:
    samples = []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        name = class_dir.name.upper()
        if name not in CLASS_TO_IDX:
            continue
        idx  = CLASS_TO_IDX[name]
        imgs = [p for p in class_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMG_EXTS]
        if max_per_class is not None:
            imgs = imgs[:max_per_class]
        samples.extend((p, idx) for p in imgs)
    return samples


def download(name: str, max_per_class: int | None = None) -> list:
    kaggle_id = DATASETS[name]
    print(f"  Downloading {kaggle_id} ...")
    base = Path(kagglehub.dataset_download(kaggle_id))
    root = find_dataset_root(base)
    samples = collect_samples(root, max_per_class)
    n_cls = len({s[1] for s in samples})
    cap_note = f"  (capped at {max_per_class}/class)" if max_per_class else ""
    print(f"  [{name}] {len(samples)} images, {n_cls} classes  root: {root.name}{cap_note}")
    return samples


# ── model loading utilities ───────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, input_size: int = LATENT_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(),
            nn.Linear(256, 128),        nn.ReLU(),
            nn.Linear(128, 64),         nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def load_pretrained_encoder() -> Encoder:
    enc = Encoder(latent_dim=LATENT_DIM).to(DEVICE)
    full_sd = torch.load(AE_PATH, map_location=DEVICE, weights_only=True)
    enc_sd = {k[len("encoder."):]: v
              for k, v in full_sd.items() if k.startswith("encoder.")}
    enc.load_state_dict(enc_sd)
    enc.eval()
    return enc


def load_pretrained_mlp() -> MLP:
    mlp = MLP().to(DEVICE)
    mlp.load_state_dict(torch.load(MLP_PATH, map_location=DEVICE, weights_only=True))
    mlp.eval()
    return mlp


def fresh_mlp() -> MLP:
    return MLP().to(DEVICE)


def load_autoencoder_for_finetuning() -> Autoencoder:
    ae = Autoencoder(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
    sd = torch.load(AE_BEST_PATH, map_location=DEVICE, weights_only=True)
    ae.load_state_dict(sd)

    for p in ae.parameters():
        p.requires_grad = False

    for name in ("block3", "pool3", "block4", "pool4", "fc1", "drop", "fc2", "bn_z"):
        module = getattr(ae.encoder, name)
        for p in module.parameters():
            p.requires_grad = True

    for p in ae.classifier.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in ae.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in ae.parameters())
    print(f"  AE trainable params: {trainable:,} / {total:,}")
    return ae


# ── shared evaluation loop and CSV output ────────────────────────────────────

@torch.no_grad()
def evaluate(encoder: Encoder, mlp: MLP, samples: list) -> tuple:
    encoder.eval()
    mlp.eval()
    loader = DataLoader(ImageSamples(samples, eval_tf),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    per_class: dict = {}
    correct = 0
    total   = 0

    for imgs, labels in loader:
        imgs  = imgs.to(DEVICE)
        z     = encoder(imgs)
        preds = mlp(z).argmax(dim=1).cpu()
        for pred, label in zip(preds, labels):
            p, l = int(pred), int(label)
            if l not in per_class:
                per_class[l] = [0, 0]
            per_class[l][1] += 1
            if p == l:
                per_class[l][0] += 1
                correct += 1
            total += 1

    return correct, per_class, total


def save_csv(correct: int, per_class: dict, total: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [["class", "total", "correct", "acc_pct"]]
    for idx, name in enumerate(CLASS_NAMES):
        if idx not in per_class:
            continue
        c, t = per_class[idx]
        rows.append([name, t, c, f"{c/t*100:.2f}"])
    rows.append(["TOTAL", total, correct, f"{correct/total*100:.2f}"])
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  Saved -> {path}")


def print_report(label: str, correct: int, per_class: dict, total: int) -> None:
    print(f"\n  {label}")
    print(f"  Overall: {correct}/{total} = {correct/total*100:.2f}%")
    print(f"  {'Class':>10}  {'Acc':>7}")
    for idx, name in enumerate(CLASS_NAMES):
        if idx not in per_class:
            continue
        c, t = per_class[idx]
        print(f"  {name:>10}  {c/t*100:6.1f}%")


# ── zero-shot mode ────────────────────────────────────────────────────────────

def run_zeroshot(source: str, target: str) -> None:
    # source is unused — pretrained models are fixed regardless of source dataset
    label    = f"AE+MLP zero-shot | tested on {target}"
    out_path = OUT_DIR_ZEROSHOT / f"aemplp_zeroshot_test_{target}.csv"

    print(f"\n{'='*60}")
    print(f"  MODE: zero-shot  |  target: {target}")
    print(f"{'='*60}")

    print("\n[1] Downloading target dataset ...")
    test_samples = download(target)

    print("\n[2] Loading pretrained encoder + MLP ...")
    encoder = load_pretrained_encoder()
    mlp     = load_pretrained_mlp()

    print("\n[3] Evaluating ...")
    correct, per_class, total = evaluate(encoder, mlp, test_samples)
    print_report(label, correct, per_class, total)
    save_csv(correct, per_class, total, out_path)


# ── MLP-only retrain mode ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_latents(encoder: Encoder, samples: list) -> tuple:
    encoder.eval()
    loader = DataLoader(ImageSamples(samples, eval_tf),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_z, all_y = [], []
    for imgs, labels in loader:
        z = encoder(imgs.to(DEVICE)).cpu().numpy()
        all_z.append(z)
        all_y.append(labels.numpy())
    return np.concatenate(all_z), np.concatenate(all_y)


class LatentDataset(Dataset):
    def __init__(self, z: np.ndarray, y: np.ndarray):
        self.z = torch.from_numpy(z).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.z[idx], self.y[idx]


def train_mlp(z_train: np.ndarray, y_train: np.ndarray,
              epochs: int = 20, lr: float = 5e-4) -> MLP:
    dataset = LatentDataset(z_train, y_train)
    n_val   = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    mlp       = fresh_mlp()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_sd       = None
    patience      = 5
    no_improve    = 0

    for epoch in range(epochs):
        mlp.train()
        for z_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(mlp(z_batch.to(DEVICE)), y_batch.to(DEVICE))
            loss.backward()
            optimizer.step()

        mlp.eval()
        val_loss = 0.0
        with torch.no_grad():
            for z_batch, y_batch in val_loader:
                val_loss += criterion(mlp(z_batch.to(DEVICE)),
                                      y_batch.to(DEVICE)).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_sd       = {k: v.clone() for k, v in mlp.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stop at epoch {epoch+1}  val_loss={val_loss:.4f}")
                break

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:3d}  val_loss={val_loss:.4f}")

    if best_sd is not None:
        mlp.load_state_dict(best_sd)
    return mlp


def run_mlp_only(source: str, target: str) -> None:
    label    = f"AE+MLP MLP-retrain | train on {source} -> test on {target}"
    out_path = OUT_DIR_FINETUNE / f"aemplp_mlpretrain_{source}_test_{target}.csv"

    print(f"\n{'='*60}")
    print(f"  MODE: mlp_only  |  {source} -> {target}")
    print(f"{'='*60}")

    print("\n[1] Downloading datasets ...")
    train_samples = download(source)
    test_samples  = download(target)

    print("\n[2] Loading pretrained encoder (frozen) ...")
    encoder = load_pretrained_encoder()

    print("\n[3] Extracting latents from source dataset ...")
    z_train, y_train = extract_latents(encoder, train_samples)
    print(f"    Latent shape: {z_train.shape}")

    print("\n[4] Training fresh MLP on source latents ...")
    mlp = train_mlp(z_train, y_train)

    print("\n[5] Evaluating on target dataset ...")
    correct, per_class, total = evaluate(encoder, mlp, test_samples)
    print_report(label, correct, per_class, total)
    save_csv(correct, per_class, total, out_path)


# ── encoder+MLP fine-tune mode ────────────────────────────────────────────────

def finetune_encoder(ae: Autoencoder, samples: list,
                     epochs: int = 30, lr: float = 1e-4) -> Autoencoder:
    n_val   = max(1, int(0.2 * len(samples)))
    n_train = len(samples) - n_val

    torch.manual_seed(42)
    shuffled      = torch.randperm(len(samples)).tolist()
    val_indices   = shuffled[:n_val]
    train_indices = shuffled[n_val:]

    train_samples_split = [samples[i] for i in train_indices]
    val_samples_split   = [samples[i] for i in val_indices]

    train_ds = ImageSamples(train_samples_split, train_tf)
    val_ds   = ImageSamples(val_samples_split,   eval_tf)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, ae.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_loss = float("inf")
    best_enc_sd   = None
    patience      = 8
    no_improve    = 0

    for epoch in range(epochs):
        ae.train()
        # Keep frozen BN layers in eval to preserve their running statistics
        ae.encoder.block1.eval()
        ae.encoder.pool1.eval()
        ae.encoder.block2.eval()
        ae.encoder.pool2.eval()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            z      = ae.encoder(imgs)
            logits = ae.classifier(z)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        ae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                z      = ae.encoder(imgs)
                logits = ae.classifier(z)
                val_loss += criterion(logits, labels).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save encoder only — ae.classifier is discarded after fine-tuning;
            # only the encoder weights matter downstream.
            best_enc_sd   = {k: v.clone() for k, v in ae.encoder.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stop at epoch {epoch+1}  val_loss={val_loss:.4f}")
                break

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:3d}  val_loss={val_loss:.4f}")

    if best_enc_sd is not None:
        ae.encoder.load_state_dict(best_enc_sd)
    ae.eval()
    return ae


def run_full_finetune(source: str, target: str) -> None:
    label    = f"AE+MLP full-finetune | train on {source} -> test on {target}"
    out_path = OUT_DIR_FINETUNE / f"aemplp_fullfinetune_{source}_test_{target}.csv"

    print(f"\n{'='*60}")
    print(f"  MODE: full  |  {source} -> {target}")
    print(f"{'='*60}")

    print("\n[1] Downloading datasets ...")
    train_samples = download(source, max_per_class=FINETUNE_MAX_PER_CLASS.get(source))
    test_samples  = download(target)

    print("\n[2] Loading AE for fine-tuning ...")
    ae = load_autoencoder_for_finetuning()

    print("\n[3] Fine-tuning encoder on source dataset ...")
    ae = finetune_encoder(ae, train_samples)

    print("\n[4] Extracting latents from source with fine-tuned encoder ...")
    ae.encoder.eval()
    z_train, y_train = extract_latents(ae.encoder, train_samples)
    print(f"    Latent shape: {z_train.shape}")

    print("\n[5] Training fresh MLP on adapted latents ...")
    mlp = train_mlp(z_train, y_train)

    print("\n[6] Evaluating on target dataset ...")
    correct, per_class, total = evaluate(ae.encoder, mlp, test_samples)
    print_report(label, correct, per_class, total)
    save_csv(correct, per_class, total, out_path)


# ── argument parsing and main entrypoint ──────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="AE+MLP cross-dataset evaluation")
    p.add_argument(
        "--mode",
        choices=["zeroshot", "mlp_only", "full", "all"],
        default="all",
    )
    p.add_argument("--source", choices=list(DATASETS), default=None)
    p.add_argument("--target", choices=list(DATASETS), default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    t0 = time.time()
    print(f"Device: {DEVICE}")

    if args.mode == "all":
        run_zeroshot(source="danrasband", target="ayuraj")
        run_zeroshot(source="ayuraj",     target="danrasband")
        run_mlp_only(source="danrasband", target="ayuraj")
        run_mlp_only(source="ayuraj",     target="danrasband")
        run_full_finetune(source="danrasband", target="ayuraj")
        run_full_finetune(source="ayuraj",     target="danrasband")
    elif args.mode == "zeroshot":
        target = args.target or "ayuraj"
        run_zeroshot(source=args.source or "danrasband", target=target)
    elif args.mode == "mlp_only":
        if not args.source or not args.target:
            print("--source and --target required for mlp_only mode")
            sys.exit(1)
        run_mlp_only(args.source, args.target)
    elif args.mode == "full":
        if not args.source or not args.target:
            print("--source and --target required for full mode")
            sys.exit(1)
        run_full_finetune(args.source, args.target)

    print(f"\nTotal time: {time.time()-t0:.1f}s")
