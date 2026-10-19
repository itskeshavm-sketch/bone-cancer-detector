"""Calibrate an uncertainty band for the demo (V12 = A12 gate + B12 specialist).

Runs the demo's own single-view inference path over the 564 test films, records
(gate probability, benign/malignant probabilities) per film, then reports where
errors concentrate in the confidence distribution so the "uncertain" band can be
set from measurement instead of guesswork.

Usage:  python calibrate_uncertainty.py
Writes: checkpoints/uncertainty_calibration.json
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b3, efficientnet_b6
from PIL import Image

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NAMES = {0: "benign", 1: "malignant", 2: "normal"}


def load(path, maker, drop=0.3):
    ckpt = torch.load(path, map_location=device)
    m = maker(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(drop),
                                nn.Linear(m.classifier[1].in_features, len(ckpt["classes"])))
    m.load_state_dict(ckpt["model"])
    return m.to(device).eval(), ckpt


modelA, ckA = load("checkpoints/stageA12.pt", efficientnet_b3)
modelB, ckB = load("checkpoints/stageB12.pt", efficientnet_b6)
IA, IB = int(ckA.get("img", 300)), int(ckB.get("img", 456))
print("A:", ckA["classes"], ckA.get("acc"), "| B:", ckB["classes"], ckB.get("acc"),
      "| device:", device, flush=True)


def tf(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


tfA, tfB = tf(IA), tf(IB)

rows = []
for cls in ["benign", "malignant", "normal"]:
    cidx = {"benign": 0, "malignant": 1, "normal": 2}[cls]
    for p in sorted(Path(f"data/test/{cls}").glob("*")):
        pil = Image.open(p).convert("RGB")
        with torch.no_grad():
            pa = F.softmax(modelA(tfA(pil).unsqueeze(0).to(device)), 1)[0]
            p_tumor = float(pa[1])
            if p_tumor < 0.5:
                pred, pb, pm = 2, 0.0, 0.0
            else:
                pb_all = F.softmax(modelB(tfB(pil).unsqueeze(0).to(device)), 1)[0]
                pb, pm = float(pb_all[0]), float(pb_all[1])
                pred = 0 if pb > pm else 1
        top = max(pb, pm) if pred != 2 else (1.0 - p_tumor)
        rows.append({"file": str(p), "truth": cidx, "pred": pred,
                     "p_tumor": p_tumor, "p_benign": pb, "p_malignant": pm,
                     "top_conf": top, "margin": abs(pb - pm) if pred != 2 else None,
                     "correct": pred == cidx})
        if len(rows) % 50 == 0:
            print("...", len(rows), flush=True)

n = len(rows)
acc = sum(r["correct"] for r in rows) / n
print(f"\nsingle-view acc: {sum(r['correct'] for r in rows)}/{n} = {acc:.4f}")

# ---- error vs confidence distribution ----
def bucket(lo, hi):
    sel = [r for r in rows if lo <= r["top_conf"] < hi]
    if not sel:
        return (0, 0, 0.0, 0.0)
    errs = sum(1 for r in sel if not r["correct"])
    return (len(sel), errs, errs / len(sel), len(sel) / n)

print("\nconf band        n     errors   err-rate   share")
for lo, hi in [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95),
               (0.95, 0.99), (0.99, 1.01)]:
    cnt, errs, rate, share = bucket(lo, hi)
    print(f"[{lo:.2f},{hi:.2f})  {cnt:4d}  {errs:5d}   {rate:6.1%}   {share:5.1%}")

# ---- candidate band thresholds ----
print("\nthreshold  flagged  of_which_err  err_capture  correct_flagged  net_benefit")
cands = []
for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    flagged = [r for r in rows if r["top_conf"] < t]
    errs = [r for r in flagged if not r["correct"]]
    total_errs = sum(1 for r in rows if not r["correct"])
    capture = len(errs) / total_errs if total_errs else 0
    net = len(errs) - (len(flagged) - len(errs))
    cands.append({"t": t, "flagged": len(flagged), "errs_in_band": len(errs),
                   "err_capture": capture, "correct_flagged": len(flagged) - len(errs),
                   "net_benefit": net})
    print(f"  {t:.2f}     {len(flagged):5d}      {len(errs):4d}      {capture:6.1%}"
          f"        {len(flagged) - len(errs):5d}         {net:+5d}")

# ---- margin-based band (specialist only) ----
tum = [r for r in rows if r["pred"] != 2]
print("\nspecialist-only margin band (n=%d tumors)" % len(tum))
for t in [0.10, 0.15, 0.20, 0.25, 0.30]:
    flagged = [r for r in tum if r["margin"] < t]
    errs = [r for r in flagged if not r["correct"]]
    total_errs = sum(1 for r in tum if not r["correct"])
    print(f"  margin<{t:.2f}: flagged {len(flagged):3d} errs {len(errs):3d} "
          f"capture {len(errs)/max(total_errs,1):5.1%} net {len(errs)-(len(flagged)-len(errs)):+4d}")

out = {"n": n, "single_view_acc": acc,
       "candidates": cands, "rows": rows}
Path("checkpoints/uncertainty_calibration.json").write_text(json.dumps(out, indent=1))
print("\nwrote checkpoints/uncertainty_calibration.json")