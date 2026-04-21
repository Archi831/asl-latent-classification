"""
Run ASL inference on a video file or webcam and save an annotated output video.

Usage:
    python predict_video.py --video path/to/asl_demo.mp4
    python predict_video.py --video 0                        # webcam
    python predict_video.py --video demo.mp4 --pipeline both
    python predict_video.py --video demo.mp4 --output out.mp4 --skip 2
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
MODELS_DIR = HERE / "models"

CNN_PATH     = MODELS_DIR / "best_cnn_asl39.pth"
ENCODER_PATH = MODELS_DIR / "encoder_ae_asl39_ld256.pth"
MLP_PATH     = MODELS_DIR / "mlp_classifier.pkl"

LATENT_DIM  = 256
NUM_CLASSES = 39
ROI_SIZE    = 200   # pixels, square ROI in frame centre

CLASS_NAMES = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'nothing','space','unknown',
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

preprocess = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------
def load_cnn(device):
    from cnn_arch import build_classifier
    model = build_classifier(num_classes=NUM_CLASSES, head="standard").to(device)
    model.load_state_dict(torch.load(CNN_PATH, map_location=device, weights_only=True))
    model.eval()
    return model


def load_encoder(device):
    from autoencoder_arch import Encoder
    enc = Encoder(latent_dim=LATENT_DIM).to(device)
    enc.load_state_dict(torch.load(ENCODER_PATH, map_location=device, weights_only=True))
    enc.eval()
    return enc


def load_mlp():
    import joblib
    return joblib.load(MLP_PATH)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def frame_to_tensor(roi_bgr) -> torch.Tensor:
    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return preprocess(pil)


@torch.no_grad()
def predict_cnn(model, tensor, device):
    logits = model(tensor.unsqueeze(0).to(device))
    probs  = torch.softmax(logits, dim=1)[0]
    top3   = probs.topk(3)
    return [(CLASS_NAMES[i], probs[i].item()) for i in top3.indices]


@torch.no_grad()
def predict_ae_mlp(encoder, mlp, tensor, device):
    z = encoder(tensor.unsqueeze(0).to(device)).cpu().numpy()
    label = mlp.predict(z)[0]
    if isinstance(label, (int, np.integer)):
        name = CLASS_NAMES[int(label)]
    else:
        name = str(label)
    proba = mlp.predict_proba(z)[0]
    idx   = CLASS_TO_IDX.get(name, 0)
    conf  = float(proba[idx]) if idx < len(proba) else 0.0
    return name, conf


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------
def draw_predictions(frame, cnn_results, ae_result=None):
    h, w = frame.shape[:2]
    x1 = w // 2 - ROI_SIZE // 2
    y1 = h // 2 - ROI_SIZE // 2
    x2 = x1 + ROI_SIZE
    y2 = y1 + ROI_SIZE

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "Place hand in box", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    # CNN predictions (left column)
    cv2.putText(frame, "CNN:", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    for i, (label, prob) in enumerate(cnn_results):
        color = (0, 255, 255) if i == 0 else (180, 180, 180)
        cv2.putText(frame, f"  {label}  {prob*100:.1f}%", (10, 58 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # AE+MLP prediction (right column)
    if ae_result is not None:
        label, conf = ae_result
        cv2.putText(frame, "AE+MLP:", (w - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        cv2.putText(frame, f"  {label}  {conf*100:.1f}%", (w - 200, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

    return frame, (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading CNN...")
    cnn_model = load_cnn(device)

    encoder = None
    mlp     = None
    use_mlp = args.pipeline in ("ae_mlp", "both")
    if use_mlp:
        if not MLP_PATH.exists():
            print(f"[warn] MLP not found at {MLP_PATH}, falling back to CNN only.")
            use_mlp = False
        else:
            print("Loading AE encoder + MLP...")
            encoder = load_encoder(device)
            mlp     = load_mlp()

    source = 0 if args.video == "0" else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Could not open video source: {args.video}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Saving annotated video to: {args.output}")

    frame_idx   = 0
    cnn_results = []
    ae_result   = None

    print("Processing frames... (press Q to stop early)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference every `skip` frames to keep it fast
        if frame_idx % (args.skip + 1) == 0:
            h, w = frame.shape[:2]
            x1 = w // 2 - ROI_SIZE // 2
            y1 = h // 2 - ROI_SIZE // 2
            roi = frame[y1:y1+ROI_SIZE, x1:x1+ROI_SIZE]

            tensor = frame_to_tensor(roi)
            cnn_results = predict_cnn(cnn_model, tensor, device)

            if use_mlp and encoder and mlp:
                ae_result = predict_ae_mlp(encoder, mlp, tensor, device)

        frame, _ = draw_predictions(frame, cnn_results, ae_result if use_mlp else None)

        if writer:
            writer.write(frame)

        cv2.imshow("ASL Real-World Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Done. Processed {frame_idx} frames.")


def main():
    parser = argparse.ArgumentParser(description="ASL video inference demo")
    parser.add_argument("--video",    required=True,
                        help="Path to video file, or '0' for webcam")
    parser.add_argument("--pipeline", default="both",
                        choices=["cnn", "ae_mlp", "both"],
                        help="Which pipeline(s) to run (default: both)")
    parser.add_argument("--output",   default="",
                        help="Save annotated video to this path (optional)")
    parser.add_argument("--skip",     type=int, default=1,
                        help="Run inference every N+1 frames (0=every frame, 1=every other, etc.)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
