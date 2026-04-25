# Real-World ASL Evaluation

Standalone module for evaluating CNN and AE+MLP pipelines on real-world footage.

## Models required in `models/`

| File | Status | Source |
|------|--------|--------|
| `best_cnn_asl39.pth` | ✅ present | copied from `cnn_classifier/models/best_cnn_classifier_asl39_ab5_cosine.pth` |
| `encoder_ae_asl39_ld256.pth` | ✅ present | copied from `autoencoder/models/encoder_ae_asl39_ld256.pth` |
| `mlp_classifier.pkl` | ❌ **needed** | ask Lilia to add `joblib.dump(model, "mlp_classifier.pkl")` to `mlp_asl.py` and share the file |

## Dependencies

```
pip install torch torchvision opencv-python pillow joblib scikit-learn
```

## Usage

### 1. Batch evaluation against a second dataset

```bash
python evaluate_realworld.py --dataset /path/to/dataset
```

Dataset must be organised as `dataset/ClassName/image.jpg`. Class folder names must match ASL39 labels (`A`–`Z`, `0`–`9`, `nothing`, `space`, `unknown`).

Outputs a comparison table to stdout and saves `realworld_results.csv`.

```bash
# CNN only (if MLP not yet available)
python evaluate_realworld.py --dataset /path/to/dataset --no-mlp
```

### 2. Video demo

```bash
python predict_video.py --video path/to/demo.mp4 --output demo_annotated.mp4
python predict_video.py --video 0                          # live webcam
python predict_video.py --video demo.mp4 --pipeline cnn    # CNN only
```

`--skip N` runs inference every N+1 frames (default 1 = every other frame). Set to 0 for every frame.
