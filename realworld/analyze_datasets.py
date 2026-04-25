"""
Dataset analysis script for evaluating external ASL datasets before inclusion in experiments.

Produces:
  - Quantitative stats: class distribution, image properties, domain distance
  - Sample images saved for visual inspection
  - Markdown checklist for human review

Usage:
    python realworld/analyze_datasets.py --dataset ayuraj
    python realworld/analyze_datasets.py --dataset danrasband
"""
import argparse
import math
import random
import sys
from pathlib import Path

import kagglehub
import numpy as np
from PIL import Image

HERE = Path(__file__).parent.parent  # repo root
OUT_DIR = HERE / "realworld" / "dataset_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "0","1","2","3","4","5","6","7","8","9",
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "nothing","space","unknown",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DATASETS = {
    "ayuraj":     ("ayuraj/asl-dataset",            None),
    "danrasband": ("danrasband/asl-alphabet-test",  None),
}

# ASL39 training set approximate pixel stats (grayscale, normalized to [0,1], 64x64)
# Computed from the full training split: mean ~0.555, std ~0.238
TRAIN_MEAN = 0.555
TRAIN_STD  = 0.238

SAMPLES_PER_CLASS = 5  # images to save per class for visual inspection
MAX_STAT_IMAGES = 2000  # images to sample for computing pixel stats


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


def collect_by_class(dataset_root: Path) -> dict[str, list[Path]]:
    """Returns {class_name: [image_paths]}. Class names uppercase, matched to CLASS_NAMES."""
    by_class: dict[str, list[Path]] = {}
    skipped = []
    for class_dir in sorted(dataset_root.iterdir()):
        if not class_dir.is_dir():
            continue
        name = class_dir.name.upper()
        if name not in CLASS_TO_IDX:
            skipped.append(class_dir.name)
            continue
        imgs = [p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
        if imgs:
            by_class[name] = imgs
    if skipped:
        print(f"  [skip unrecognized dirs] {skipped[:10]}{'...' if len(skipped)>10 else ''}")
    return by_class


def image_native_stats(path: Path) -> tuple[tuple[int,int], int]:
    """Returns (width, height), n_channels from the raw image file."""
    try:
        with Image.open(path) as img:
            w, h = img.size
            n_ch = len(img.getbands())
            return (w, h), n_ch
    except Exception:
        return (0, 0), 0


def pixel_stats_sample(paths: list[Path], n: int = MAX_STAT_IMAGES) -> tuple[float, float]:
    """Sample up to n images, convert to grayscale 64x64 [0,1], return mean and std."""
    sample = paths if len(paths) <= n else random.sample(paths, n)
    values = []
    for p in sample:
        try:
            with Image.open(p) as img:
                arr = np.array(img.convert("L").resize((64, 64)), dtype=np.float32) / 255.0
                values.append(arr.mean())
        except Exception:
            continue
    if not values:
        return float("nan"), float("nan")
    arr = np.array(values)
    return float(arr.mean()), float(arr.std())


def save_sample_images(name: str, by_class: dict[str, list[Path]], n: int = SAMPLES_PER_CLASS) -> Path:
    """Save a grid of sample images (n per class) as a PNG for visual inspection."""
    classes = sorted(by_class.keys())
    n_cols = n
    n_rows = len(classes)
    cell = 64
    pad = 2
    grid_w = n_cols * (cell + pad) + pad
    grid_h = n_rows * (cell + pad) + pad + 20  # +20 for class label row

    from PIL import ImageDraw, ImageFont
    grid = Image.new("RGB", (grid_w, grid_h), (40, 40, 40))
    draw = ImageDraw.Draw(grid)

    for row, cls in enumerate(classes):
        paths = by_class[cls]
        sample = paths[:n] if len(paths) >= n else paths + [None] * (n - len(paths))
        # draw class label
        y_label = pad + row * (cell + pad)
        draw.text((pad, y_label + cell // 2 - 6), cls, fill=(220, 220, 220))
        for col, p in enumerate(sample):
            x = pad + col * (cell + pad) + 20  # offset for label
            y = y_label
            if p is not None:
                try:
                    with Image.open(p) as img:
                        thumb = img.convert("RGB").resize((cell, cell))
                        grid.paste(thumb, (x, y))
                except Exception:
                    pass

    out = OUT_DIR / f"samples_{name}.png"
    grid.save(out)
    return out


def write_checklist(name: str, report_lines: list[str]) -> Path:
    out = OUT_DIR / f"checklist_{name}.md"
    lines = [
        f"# Visual Analysis Checklist — {name}",
        "",
        "Refer to `dataset_analysis/samples_{name}.png` while filling this in.",
        "",
        "## Quantitative Summary",
        "",
        *[f"    {l}" for l in report_lines],
        "",
        "## Visual Inspection",
        "",
        "For each item, mark one option and add notes if needed.",
        "",
        "| Aspect | Assessment | Notes |",
        "|---|---|---|",
        "| Background | [ ] plain/uniform  [ ] complex/cluttered | |",
        "| Hand orientation | [ ] consistent  [ ] varied | |",
        "| Lighting | [ ] controlled  [ ] naturalistic | |",
        "| Skin tone diversity | [ ] low  [ ] high | |",
        "| Image quality | [ ] clean  [ ] noisy/blurry | |",
        "| Channel format | [ ] grayscale  [ ] RGB  [ ] RGB+depth | |",
        "",
        "## Problem Classes",
        "",
        "List any classes that look visually incompatible with the ASL39 training data:",
        "",
        "- ",
        "",
        "## Inclusion Decision",
        "",
        "Based on quantitative + visual analysis:",
        "",
        "- [ ] Include in fine-tuning experiments",
        "- [ ] Exclude — reason: ___",
        "- [ ] Include as test-only (too different to fine-tune on, but useful for evaluation)",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def analyze(name: str) -> None:
    if name not in DATASETS:
        print(f"Unknown dataset '{name}'. Choose from: {list(DATASETS)}")
        sys.exit(1)

    kaggle_id, _ = DATASETS[name]
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}  ({kaggle_id})")
    print(f"{'='*60}")

    print("  Downloading...")
    base = Path(kagglehub.dataset_download(kaggle_id))
    root = find_dataset_root(base)
    print(f"  Dataset root: {root}")

    by_class = collect_by_class(root)
    all_paths = [p for paths in by_class.values() for p in paths]
    n_classes = len(by_class)
    n_images  = len(all_paths)

    print(f"  Classes found: {n_classes} / 39")
    print(f"  Total images:  {n_images}")

    # per-class counts
    counts = {cls: len(paths) for cls, paths in by_class.items()}
    missing = [c for c in CLASS_NAMES if c not in by_class]
    min_count = min(counts.values()) if counts else 0
    max_count = max(counts.values()) if counts else 0
    avg_count = n_images / n_classes if n_classes else 0

    # native image properties (sample up to 50 images)
    stat_sample = random.sample(all_paths, min(50, len(all_paths)))
    resolutions = []
    channels    = []
    for p in stat_sample:
        (w, h), ch = image_native_stats(p)
        if w > 0:
            resolutions.append((w, h))
            channels.append(ch)

    unique_res   = list(set(resolutions))
    channel_mode = max(set(channels), key=channels.count) if channels else 0

    # pixel stats on grayscale 64x64
    print("  Computing pixel statistics (grayscale 64×64)...")
    ds_mean, ds_std = pixel_stats_sample(all_paths)
    domain_dist = abs(ds_mean - TRAIN_MEAN)

    # build report lines
    report_lines = [
        f"Dataset:          {name}",
        f"Kaggle ID:        {kaggle_id}",
        f"Total images:     {n_images}",
        f"Classes present:  {n_classes} / 39",
        f"Missing classes:  {missing if missing else 'none'}",
        f"Samples/class:    min={min_count}  max={max_count}  avg={avg_count:.0f}",
        f"Native resol.:    {unique_res[:5]}{'...' if len(unique_res)>5 else ''}",
        f"Channel mode:     {channel_mode} ({'grayscale' if channel_mode==1 else 'RGB' if channel_mode==3 else 'RGBA/other'})",
        f"Pixel mean (gs):  {ds_mean:.4f}  (train: {TRAIN_MEAN:.4f})",
        f"Pixel std  (gs):  {ds_std:.4f}  (train: {TRAIN_STD:.4f})",
        f"Domain distance:  |mean diff| = {domain_dist:.4f}",
    ]

    # per-class table
    report_lines += ["", "Per-class sample counts:"]
    for cls in CLASS_NAMES:
        cnt = counts.get(cls, 0)
        bar = "#" * min(40, cnt // 10)
        flag = "  ** MISSING **" if cnt == 0 else ("  ** LOW **" if cnt < 20 else "")
        report_lines.append(f"  {cls:>10}: {cnt:5d}  {bar}{flag}")

    # print report
    print()
    for line in report_lines:
        print(f"  {line}")

    # save report text
    report_path = OUT_DIR / f"report_{name}.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n  Report saved: {report_path}")

    # save sample images
    print("  Saving sample images...")
    grid_path = save_sample_images(name, by_class)
    print(f"  Sample grid:  {grid_path}")

    # write checklist
    checklist_path = write_checklist(name, report_lines)
    print(f"  Checklist:    {checklist_path}")

    print(f"\n  Done: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze an external ASL dataset")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS),
                        help="Dataset name to analyze")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    analyze(args.dataset)


if __name__ == "__main__":
    main()
