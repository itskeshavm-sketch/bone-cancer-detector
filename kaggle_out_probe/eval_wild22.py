"""Score V22 (A18 gate + B22 lesion-aware specialist) on the 199-film wild pool.

Sweeps the malignant decision threshold so V22 can be compared with V21 and V12
at MATCHED cancer sensitivity - the only comparison that means anything (the
benchmark and the wild pool disagreed for five straight versions because total
accuracy averages two errors that trade off).

Writes per-film records so any pair of models can be diffed later.
Usage: python eval_wild22.py [checkpoints/stageB22.pt] [600]
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b3, efficientnet_b6

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SPEC_PATH = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/stageB22.pt"
SPEC_IMG = int(sys.argv[2]) if len(sys.argv) > 2 else 600
GATE_PATH = "checkpoints/stageA18.pt"
BATCH = 16


def load(path, maker, ncls=2, drop=0.4):
    ckpt = torch.load(path, map_location="cpu")
    m = maker(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(drop), nn.Linear(m.classifier[1].in_features, ncls))
    m.load_state_dict(ckpt["model"])
    return m.eval()


mA = load(GATE_PATH, efficientnet_b3)
mB = load(SPEC_PATH, efficientnet_b6)
evA = transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor(),
                          transforms.Normalize(MEAN, STD)])
evB = transforms.Compose([transforms.Resize((SPEC_IMG, SPEC_IMG)), transforms.ToTensor(),
                          transforms.Normalize(MEAN, STD)])

files = sorted(Path("wild_test").glob("ra_*.png"))
recs = []
for start in range(0, len(files), BATCH):
    chunk = files[start:start + BATCH]
    pils = [Image.open(p).convert("RGB") for p in chunk]
    with torch.inference_mode():
        xa = torch.stack([evA(im) for im in pils])
        tumor = F.softmax(mA(xa), 1)[:, 1]
        xb = torch.stack([evB(im) for im in pils])
        pb = F.softmax(mB(xb), 1)
    for i, p in enumerate(chunk):
        truth = 1 if ("_malignant_" in p.name or p.name.startswith("ra_malignant")) else 0
        recs.append({"file": p.name, "truth": truth,
                     "tumor": float(tumor[i]),
                     "ben": float(pb[i][0]), "mal": float(pb[i][1])})
    print(f"  {min(start+BATCH, len(files))}/{len(files)}", flush=True)

n = len(recs)
print(f"\nscored {n} wild films\n", flush=True)

THRESH = [round(x, 3) for x in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                               0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]]


def sweep(thr, gate_thr=0.5):
    ok = ben_ok = ben_tot = mal_ok = mal_tot = 0
    for r in recs:
        if r["tumor"] < gate_thr:
            pred = 2
        else:
            pred = 1 if r["mal"] >= thr else 0
        good = pred == r["truth"]
        ok += good
        if r["truth"] == 0:
            ben_tot += 1
            ben_ok += good
        else:
            mal_tot += 1
            mal_ok += good
    return {"acc": ok / n, "ok": ok, "mal": mal_ok, "mal_pct": mal_ok / mal_tot,
            "ben": ben_ok, "ben_pct": ben_ok / ben_tot, "thr": thr}


rows = {str(t): sweep(t) for t in THRESH}
print(f"{'thr':>6}{'acc':>9}{'ok':>7}{'mal':>9}{'mal%':>8}{'ben':>9}{'ben%':>8}")
for t in THRESH:
    r = rows[str(t)]
    mark = "  <-- argmax" if t == 0.5 else ""
    print(f"{t:>6.2f}{r['acc']:>9.1%}{str(r['ok'])+'/'+str(n):>7}"
          f"{str(r['mal'])+'/95':>9}{r['mal_pct']:>8.1%}{str(r['ben'])+'/104':>9}"
          f"{r['ben_pct']:>8.1%}{mark}")

target = 49  # V12/V21 matched operating point: 49/95 cancers caught
cands = [(t, rows[str(t)]) for t in THRESH]
hit = [c for c in cands if c[1]["mal"] == target]
if hit:
    t, r = min(hit, key=lambda c: abs(c[0] - 0.5))
    print(f"\nMATCHED at {target}/95 cancers: thr={t:.2f} acc={r['acc']:.1%} "
          f"({r['ok']}/{n}) mal={r['mal']}/95 ben={r['ben']}/104")
else:
    t, r = min(cands, key=lambda c: abs(c[1]["mal"] - target))
    print(f"\nMATCHED at {target}/95: no exact threshold; closest thr={t:.2f} "
          f"acc={r['acc']:.1%} mal={r['mal']}/95 ben={r['ben']}/104")

t_best, r_best = max(rows.items(), key=lambda kv: kv[1]["acc"])
print(f"BEST ACHIEVABLE: acc={r_best['acc']:.1%} ({r_best['ok']}/{n}) at thr={t_best} "
      f"mal={r_best['mal']}/95 ben={r_best['ben']}/104")

out = {"spec": SPEC_PATH, "spec_img": SPEC_IMG, "n": n,
       "sweep": rows, "matched_thr": t, "matched": r, "best": r_best, "records": recs}
Path("kaggle_out_probe/wild_v22.json").write_text(json.dumps(out, indent=1))
print("\nwrote kaggle_out_probe/wild_v22.json")
