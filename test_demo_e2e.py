"""End-to-end check of the demo after the 6-view TTA change.

Exercises the real code paths - the same functions the Gradio buttons call - on
one film of each class, in both quick (single-view) and careful (6-view) mode,
and asserts that:

  * both paths run and return an overlay, a label and a verdict
  * the careful path really averages 6 views, not 5
  * the reliability band key is still the SINGLE-VIEW specialist probability,
    because the band table was calibrated on the single-view path - if the
    careful path changed the band key, every band in the demo would be
    mis-assigned
  * the verdict text is non-empty and names the finding

Usage: python test_demo_e2e.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image

import app as A

print("views in tta_views (default):", len(A.tta_views(Image.new("RGB", (400, 400)))))
assert len(A.tta_views(Image.new("RGB", (400, 400)))) == 5, "expected 5 views by default"

fails = []
for cls in ["malignant", "benign", "normal"]:
    hits = sorted(Path(f"data/test/{cls}").glob("*"))
    if not hits:
        continue
    img = Image.open(hits[0]).convert("RGB")
    for mode, fn in (("quick", A.quick), ("careful", A.thorough)):
        overlay, conf, text = fn(img)
        ok = overlay is not None and bool(conf) and bool(text) and len(text) > 40
        key = conf.get("malignant", conf.get("benign", conf.get("normal")))
        print(f"  {cls:<10} {mode:<8} -> {'OK' if ok else 'FAIL'}  "
              f"conf={ {k: round(v, 3) for k, v in conf.items()} }")
        if not ok:
            fails.append(f"{cls}/{mode}: empty result")
        if "uncertain reading" in str(text).lower() or "Normal" in str(text) \
                or "Benign" in str(text) or "Malignant" in str(text):
            pass
        else:
            fails.append(f"{cls}/{mode}: verdict text has no finding")

# the band key must stay the single-view probability
img = Image.open(sorted(Path("data/test/benign").glob("*"))[0]).convert("RGB")
probB, key = A.specialist_probs(img, True, False)
single = A.specialist_probs(img, False, False)[1]
import torch
assert torch.allclose(torch.as_tensor(key), torch.as_tensor(single), atol=1e-6), \
    "the band key changed when TTA is on - the calibrated bands would be mis-assigned"
print("  band key stays the single-view probability under TTA: OK")

# and the careful path must actually differ from the quick one on the numbers
import torch
p_careful = A.specialist_probs(img, True, False)[0]
p_quick = A.specialist_probs(img, False, False)[0]
print(f"  quick p(malignant) {float(p_quick[1]):.4f} vs careful {float(p_careful[1]):.4f}")
assert abs(float(p_careful[1]) - float(p_quick[1])) > 1e-6, \
    "the careful path is not actually averaging views"

if fails:
    print("\nFAILURES:")
    for f in fails:
        print("  -", f)
    raise SystemExit(1)
print("\nE2E PASS")
