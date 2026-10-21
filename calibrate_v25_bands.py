"""Calibrate the uncertainty band for the NEW champion (V25 = stageA18 + stageB25).

Same methodology as the V21 band so the two are comparable: single-view path,
gate 0.30, malignant threshold 0.25, over all 564 benchmark films, band edges at
0.60/0.70/0.80/0.90, flag a band when its measured error rate is >= 25%
(3x the model's overall error rate).

Writes:
  checkpoints/benchmark_rows_v25.json   per-film confidence rows
  checkpoints/uncertainty_bands_v25.json  the band table the demo will load
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b3, efficientnet_b6

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
BATCH = 16
GATE_THR, MAL_THR = 0.30, 0.25
UNCERTAIN_ERR = 0.25
EDGES = [0.0, 0.60, 0.70, 0.80, 0.90, 1.01]


def load(path, maker, ncls=2, drop=0.3):
    ckpt = torch.load(path, map_location="cpu")
    m = maker(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(drop), nn.Linear(m.classifier[1].in_features, ncls))
    m.load_state_dict(ckpt["model"])
    return m.eval()


gate = load("checkpoints/stageA18.pt", efficientnet_b3)
spec = load("checkpoints/stageB25.pt", efficientnet_b6)
evA = transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor(),
                          transforms.Normalize(MEAN, STD)])
evB = transforms.Compose([transforms.Resize((456, 456)), transforms.ToTensor(),
                           transforms.Normalize(MEAN, STD)])

rows = []
ROWS_PATH = Path("checkpoints/benchmark_rows_v25.json")
if ROWS_PATH.exists():
    try:
        rows = json.loads(ROWS_PATH.read_text())
    except Exception:
        rows = []
done_files = {r["file"] for r in rows}
print(f"resuming with {len(rows)} rows", flush=True)
for cls in ["benign", "malignant", "normal"]:
    cidx = {"benign": 0, "malignant": 1, "normal": 2}[cls]
    files = [p for p in sorted(Path(f"data/test/{cls}").glob("*"))
             if str(p) not in done_files]
    if not files:
        continue
    for start in range(0, len(files), BATCH):
        chunk = files[start:start + BATCH]
        pils = [Image.open(p).convert("RGB") for p in chunk]
        with torch.inference_mode():
            xa = torch.stack([evA(im) for im in pils])
            tumor = F.softmax(gate(xa), 1)[:, 1]
            xb = torch.stack([evB(im) for im in pils])
            pb = F.softmax(spec(xb), 1)
        for i, p in enumerate(chunk):
            tp = float(tumor[i])
            if tp < GATE_THR:
                pred, top = 2, 1.0 - tp
            else:
                pred = 1 if float(pb[i][1]) >= MAL_THR else 0
                top = max(float(pb[i][0]), float(pb[i][1]))
            rows.append({"file": str(p), "truth": cidx, "pred": pred,
                         "tumor": tp, "ben": float(pb[i][0]), "mal": float(pb[i][1]),
                         "top_conf": top, "correct": int(pred == cidx)})
    print(f"  {cls} done ({len(rows)} rows)", flush=True)
    ROWS_PATH.write_text(json.dumps(rows, indent=1))

n = len(rows)
ok = sum(r["correct"] for r in rows)
per = {0: [0, 0], 1: [0, 0], 2: [0, 0]}
for r in rows:
    per[r["truth"]][0] += r["correct"]
    per[r["truth"]][1] += 1
print(f"\nsingle-view acc: {ok}/{n} = {ok/n:.4f}")
print(f"  benign {per[0][0]}/{per[0][1]} | malignant {per[1][0]}/{per[1][1]} | "
      f"normal {per[2][0]}/{per[2][1]}")

bands = []
for i in range(len(EDGES) - 1):
    lo, hi = EDGES[i], EDGES[i + 1]
    sel = [r for r in rows if lo <= r["top_conf"] < hi]
    if not sel:
        bands.append({"lo": lo, "hi": hi, "n": 0, "accuracy": ok / n,
                      "error_rate": 1 - ok / n, "uncertain": False})
        continue
    acc = sum(r["correct"] for r in sel) / len(sel)
    bands.append({"lo": lo, "hi": hi, "n": len(sel), "accuracy": acc,
                  "error_rate": 1 - acc, "share": len(sel) / n,
                  "uncertain": (1 - acc) >= UNCERTAIN_ERR})

err_total = sum(1 for r in rows if not r["correct"])
flagged = [r for r in rows if any(
    b["uncertain"] and b["lo"] <= r["top_conf"] < b["hi"] for b in bands)]
flagged_err = sum(1 for r in flagged if not r["correct"])

out = {
    "model": "V25 (stageA18 gate + stageB25 specialist, gate 0.30, malignant 0.25)",
    "n_films": n, "overall_accuracy": ok / n, "overall_error_rate": 1 - ok / n,
    "uncertain_error_threshold": UNCERTAIN_ERR, "bands": bands,
    "flagged_count": len(flagged), "flagged_share": len(flagged) / n,
    "flagged_error_rate": (flagged_err / len(flagged)) if flagged else 0.0,
    "errors_total": err_total, "errors_caught": flagged_err,
    "error_capture": flagged_err / err_total if err_total else 0.0,
    "reliable_band": next((b for b in bands if not b["uncertain"] and b["n"] > 0), None),
    "per_class": {k: f"{v[0]}/{v[1]}" for k, v in per.items()},
}
Path("checkpoints/uncertainty_bands_v25.json").write_text(json.dumps(out, indent=1))
Path("checkpoints/benchmark_rows_v25.json").write_text(json.dumps(rows, indent=1))

print(f"\n{'band':<16}{'n':>5}{'acc':>9}{'err':>8}  uncertain")
for b in bands:
    print(f"[{b['lo']:.2f},{b['hi']:.2f})   {b['n']:>5}{b['accuracy']:>9.3f}"
          f"{b['error_rate']:>8.3f}  {'YES' if b['uncertain'] else 'no'}")
print(f"uncertain band: {len(flagged)} films ({len(flagged)/n:.1%}), err "
      f"{out['flagged_error_rate']:.1%}, captures {flagged_err}/{err_total} = "
      f"{out['error_capture']:.1%}")
print("\nwrote checkpoints/uncertainty_bands_v25.json")