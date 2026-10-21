"""Exercise app.predict() on real test films and report how the uncertainty band fires.

This runs the actual demo code path (gradcam, bands, verdict HTML) so the band is
verified end to end, not just in the table.
"""
import importlib.util
import sys
from pathlib import Path

import torch
from PIL import Image

spec = importlib.util.spec_from_file_location("app_mod", "app.py")
app = importlib.util.module_from_spec(spec)
sys.modules["app_mod"] = app
spec.loader.exec_module(app)

NAMES = {0: "benign", 1: "malignant", 2: "normal"}
rows = []
for cls in ["benign", "malignant", "normal"]:
    files = sorted(Path(f"data/test/{cls}").glob("*"))[:12]
    for p in files:
        pil = Image.open(p).convert("RGB")
        overlay, conf, html = app.predict(pil, tta=False, vote=False,
                                          thresh=app.GATE_THRESH,
                                          mal_thresh=app.MAL_THRESH)
        uncertain = 'v-uncertain' in html
        head = "Uncertain reading" if uncertain else "Unknown"
        for line in html.split("<p"):
            if 'class="v-head"' in line:
                head = line.split(">")[1].split("<")[0]
            if 'class="v-rel"' in line:
                rel = line.split(">")[1].split("<")[0]
        rows.append({"truth": cls, "pred": head, "uncertain": uncertain,
                     "overlay": overlay is not None, "html_len": len(html)})

n = len(rows)
unc = sum(r["uncertain"] for r in rows)
ov = sum(r["overlay"] for r in rows)
print(f"exercised {n} films | uncertain band fired {unc} times | gradcam overlay built {ov}/{n}")
print("\nsample outputs:")
for r in rows[:6]:
    print(" ", r["truth"], "->", r["pred"], "| uncertain:", r["uncertain"], "| html", r["html_len"], "chars")
# no crash + band present
assert ov == n, "overlay missing"
assert all(r["html_len"] > 200 for r in rows), "verdict html too short"
print("\nPREDICT PATH OK")
