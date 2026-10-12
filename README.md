# 🦴 Bone Cancer Detection AI

Two-stage X-ray classifier: **normal / benign tumor / malignant tumor**, with Grad-CAM heatmaps showing where the model looked.

## How it works

1. **Stage A — gate** (EfficientNet-B3, 300px): tumor vs normal. Val accuracy 96.2%.
2. **Stage B — specialist** (EfficientNet-B6, 456px): benign vs malignant. Val accuracy 89.5%.
3. Test-time augmentation (5 views) at eval.

Two models beat one because each learns a single easy question.

## Results (held-out test, 564 X-rays, TTA)

| Version | Test acc | Notes |
|---|---|---|
| V7 (B3+B5) | 87.9% | first strong two-stage |
| V10 (B3+B5, SWA rejected) | 88.5% | MixUp+EMA tried, dropped |
| V11 (B3+B6) | 89.4% | best gate yet at the time |
| **V12 (B3+B6, seed 12)** | **89.5%** | champion |
| V13 (B3+B7, focal loss) | 85.8% | focal + lower threshold overfit; recipe rejected |

Per-class (V11): malignant P/R 89.4%, benign F1 85.1%, normal F1 93.3%.

## Launch demo

```bash
pip install -r requirements.txt
python app.py --share
```

Tick *Accurate mode* for the 5-view vote that matches the 89.4% number.

## ⚠️ Disclaimer

**Proof-of-concept for educational/hackathon purposes only — NOT for clinical use.**
The model is wrong ~1 in 10 times. Never use it for patient decisions; always consult a medical professional.
