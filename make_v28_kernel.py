"""Derive the V28 kernel from train_v26.py.

Replaces the module docstring wholesale (whitespace-exact string matching on a
multi-line block is a trap - the V26 source has no leading indent on some lines)
and then applies single-token substitutions that are unambiguous.

Prints a diff summary so the change is auditable before the push.

Usage: python make_v28_kernel.py
"""
from pathlib import Path

SRC = Path("kaggle_gpu/train_v26.py")
DST = Path("kaggle_v28/train_v28.py")
DST.parent.mkdir(parents=True, exist_ok=True)

src = SRC.read_text(encoding="utf-8")
lines = src.splitlines(keepends=True)

NEW_DOC = '''"""Kaggle GPU V28: V27's exact recipe with the non-radiographs removed from the
harvest, so the only variable is whether the harvested data is clean.

V25 (champion): 159 real images x 6 repeat =   954 items vs 2,736 main -> 25.9% real
V27 (previous): 243 real images x 6 repeat = 1,458 items, of which 43 are NOT
                radiographs - CT slices, MRI, pathology micrographs and bone
                scans - each carrying a page-level bone-tumour label (by-eye audit)
V28 (this run): 200 real images x 6 repeat = 1,200 items, ZERO non-radiographs

Same init (stageB21), lr 2e-5, 6 epochs, B6 456px, A18 gate reused, the same
78-film held-out slice and the same repeat factor as V27. The ONLY difference is
that 43 poisoned images are no longer in the training set.

If V28 beats V27, the contamination was the problem and the harvest road is open
again. If V28 ties V25, the harvest adds nothing beyond a handful of clean images
and the data-scaling road is closed.

The label lookup uses ONE rule for images and labels (see wild_key) - the V25
version shipped a mismatched-prefix bug that silently trained on zero wild images.
Any unresolved split entry now ABORTS the run.
"""
'''

# the docstring is lines 0..17 (index), ending at the first line that is exactly '"""\n'
end = next(i for i, l in enumerate(lines) if l.rstrip("\n") == '"""' and i > 0)
out = NEW_DOC + "".join(lines[end + 1:])

TOKENS = [
    ("random.seed(26)", "random.seed(28)"),
    ('TMP = Path("/tmp/kwork26")', 'TMP = Path("/tmp/kwork28")'),
    ("WILD_REPEAT = 4", "WILD_REPEAT = 6"),
    ('"specB26.pt"', '"specB28.pt"'),
    ('desc=f"B26 {ep}/{EPOCHS}"', 'desc=f"B28 {ep}/{EPOCHS}"'),
    ('print(f"B26 Ep{ep}: train=', 'print(f"B28 Ep{ep}: train='),
    ('print(f"B26 best btxrd val: {best:.4f}"', 'print(f"B28 best btxrd val: {best:.4f}"'),
    ('"v": 26, "init": "stageB21"', '"v": 28, "init": "stageB21"'),
    ('"harvested_added": split.get("harvested_added"),',
     '"harvested_added": split.get("harvested_added"),\n'
     '        "harvested_rejected": split.get("harvested_rejected"),'),
]
for old, new in TOKENS:
    if old not in out:
        raise SystemExit(f"TOKEN NOT FOUND: {old!r}")
    out = out.replace(old, new)

DST.write_text(out, encoding="utf-8")

import ast
ast.parse(out)
checks = {
    "WILD_REPEAT = 6": "WILD_REPEAT = 6" in out,
    "specB28.pt": "specB28.pt" in out,
    "v=28": '"v": 28' in out,
    "kwork28": "kwork28" in out,
    "seed 28": "random.seed(28)" in out,
    "harvested_rejected logged": "harvested_rejected" in out,
    "old 'Kaggle GPU V26' gone": "Kaggle GPU V26" not in out,
    "old specB26 gone": "specB26" not in out,
    "old WILD_REPEAT = 4 gone": "WILD_REPEAT = 4" not in out,
    "old v26 metric gone": '"v": 26' not in out,
}
for k, v in checks.items():
    print(f"  {'OK ' if v else 'FAIL'} {k}")
assert all(checks.values()), "kernel transformation incomplete"

print(f"\nwrote {DST}  (syntax OK)")
print(f"  train items  = 2736 main + {200 * 6} real  -> "
      f"{200 * 6 / (200 * 6 + 2736):.1%} real")
