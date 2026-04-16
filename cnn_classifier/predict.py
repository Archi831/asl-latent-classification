# cnn_classifier/predict.py
# Real-time ASL sign prediction from webcam.
# Press SPACE to capture and predict, Q to quit.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "autoencoder"))

import cv2
import torch
import numpy as np
from torchvision import transforms

from model import build_classifier

MODELS_DIR  = Path(__file__).parent / "models"
MODEL_PATH  = MODELS_DIR / "best_cnn_classifier.pth"
CLASS_NAMES = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
               'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
               'del','nothing','space']

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),   # [0, 1]
])

def predict(model, frame, device):
    tensor = transform(frame).unsqueeze(0).to(device)   # (1, 1, 64, 64)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        top3   = probs.topk(3)
    results = [(CLASS_NAMES[i], probs[i].item()) for i in top3.indices]
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_classifier().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")
    print("Controls: SPACE = capture & predict  |  Q = quit")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    last_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw ROI box in center (200x200)
        h, w = frame.shape[:2]
        x1, y1 = w // 2 - 100, h // 2 - 100
        x2, y2 = w // 2 + 100, h // 2 + 100
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Place hand in box", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # Show last predictions
        for i, (label, prob) in enumerate(last_results):
            text  = f"#{i+1}: {label}  {prob*100:.1f}%"
            color = (0, 255, 255) if i == 0 else (200, 200, 200)
            cv2.putText(frame, text, (10, 40 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(frame, "SPACE=predict  Q=quit", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("ASL Classifier", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            roi = frame[y1:y2, x1:x2]
            last_results = predict(model, roi, device)
            print("Prediction:", " | ".join(f"{l} {p*100:.1f}%" for l, p in last_results))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
