"""Derive V30 from train_v29.py: same 9-class addition, but the real-data share is
HELD CONSTANT.

Why: adding in-distribution images to the base pool dilutes the real radiographs.
  V25: base 3,367 + real 954  -> real share 22.1%
  V29: base 3,367 + N + real 954 -> real share falls as N grows
The harvest experiments (V26, V27, V28) showed the model is sensitive to that
mix, so a V29 win could be masked by dilution and a V29 loss could be caused by
it. V30 removes the confound: it picks the repeat factor that restores V25's
22.1% share, so the ONLY thing that differs from V25 is "the base pool now also
contains fine-labelled 9-class images".

Usage: python make_v30_kernel.py
"""
from pathlib import Path

SRC = Path("kaggle_v29/train_v29.py")
DST = Path("kaggle_v30/train_v30.py")
DST.parent.mkdir(parents=True, exist_ok=True)
src = SRC.read_text(encoding="utf-8")
lines = src.splitlines(keepends=True)

NEW_DOC = '''"""Kaggle GPU V30: the 9-class BTXRD release, with the real-data share pinned to
V25's value so the mix cannot confound the result.

  V25: base 3,367 + real 954            -> real share 22.1%
  V29: base 3,367 + N + real 954        -> real share falls as N grows (dilution)
  V30: base 3,367 + N + real (repeat r) -> real share pinned back to 22.1%

Everything else is V29's, which is V25's recipe plus the 9-class images. The
repeat factor is computed at runtime from the actual base-pool size, so the mix
is identical no matter how many 9-class images actually land.

Same safety: any 9-class image matching one of the 564 benchmark test films is
dropped by md5 AND by a 16x16 perceptual signature; their own test split is never
used; label contradictions are relabelled; the run aborts if any split entry is
unresolved.
"""
'''

end = next(i for i, l in enumerate(lines) if l.rstrip("\n") == '"""' and i > 0)
out = NEW_DOC + "".join(lines[end + 1:])

TOKENS = [
    ("random.seed(29)", "random.seed(30)"),
    ('TMP = Path("/tmp/kwork29")', 'TMP = Path("/tmp/kwork30")'),
    ("WILD_REPEAT = 6          # identical to V25",
     "TARGET_REAL_SHARE = 0.2210   # exactly V25's share: 954 / (954 + 3367)"),
    ('"specB29.pt"', '"specB30.pt"'),
    ('desc=f"B29 {ep}/{EPOCHS}"', 'desc=f"B30 {ep}/{EPOCHS}"'),
    ('print(f"B29 Ep{ep}: train=', 'print(f"B30 Ep{ep}: train='),
    ('print(f"B29 best btxrd val: {best:.4f}"', 'print(f"B30 best btxrd val: {best:.4f}"'),
    ('"v": 29, "init": "stageB21"', '"v": 30, "init": "stageB21"'),
]
for old, new in TOKENS:
    if old not in out:
        raise SystemExit(f"TOKEN NOT FOUND: {old!r}")
    out = out.replace(old, new)

# --- the dynamic repeat, replacing the fixed constant ---
old_block = """    tr = MixedDataset(SUB / "train", train_wild, aug_train(IMG), WILD_REPEAT)"""
new_block = """    # Pin the real-data share to V25's value. V25 used 159 real images x 6
    # against a base of 3,367 -> 954/(954+3367) = 22.1%. The 9-class images
    # grow the base, so the repeat is recomputed to keep that share identical.
    real_items_needed = TARGET_REAL_SHARE / (1 - TARGET_REAL_SHARE) * len(tr.base)
    wild_repeat = max(1, int(round(real_items_needed / max(len(train_wild), 1))))
    print(f"base {len(list((SUB / 'train').rglob('*')))} | real needed "
          f"{real_items_needed:.0f} items | {len(train_wild)} real images -> "
          f"repeat {wild_repeat}", flush=True)

    tr = MixedDataset(SUB / "train", train_wild, aug_train(IMG), wild_repeat)"""
if old_block not in out:
    raise SystemExit("repeat block not found")
out = out.replace(old_block, new_block)

out = out.replace('"wild_repeat": WILD_REPEAT,', '"wild_repeat": wild_repeat,\n        "target_real_share": TARGET_REAL_SHARE,')
out = out.replace('f"| real share {len(tr.wild)*WILD_REPEAT/len(tr):.1%} | val {len(va)}"',
                  'f"| real share {len(tr.wild)*wild_repeat/len(tr):.1%} | val {len(va)}"')
out = out.replace('+ wild {len(tr.wild)}x{WILD_REPEAT})',
                  '+ wild {len(tr.wild)}x{wild_repeat})')

# ---- replace the byte-exact quantised near-duplicate check with the calibrated
# MAD comparison. The quantised byte signature collided across unrelated films
# (a radiograph is mostly black, so a 4-level 16x16 grid is nearly constant),
# which over-dropped good training images. Calibrated in calibrate_dedup.py:
# known same-film pairs sit at MAD <= 0.0002, unrelated pairs at a median of
# 0.255, and MAD16 < 0.02 catches 19/19 leaks while false-flagging 2 in 3000.
OLD_PSIG = '''def psig(p):
    """16x16 grayscale signature quantised to 4 levels - survives re-encoding."""
    try:
        a = np.asarray(Image.open(p).convert("L").resize((16, 16)), dtype=float) / 255.0
        return (a * 4).astype(np.int8).tobytes()
    except Exception:
        return None
'''
NEW_PSIG = '''NEAR_DUP_MAD = 0.02


def psig(p):
    """16x16 grayscale signature, NOT quantised.

    A byte-exact comparison of a QUANTISED signature collides across unrelated
    films - a radiograph is mostly black background, so a 4-level 16x16 grid is
    nearly constant and many different films hash to the same bytes. Calibrated
    in kaggle_out_probe/calibrate_dedup.py: the 19 known same-film pairs sit at
    mean-absolute-difference <= 0.0002 while unrelated pairs have a median of
    0.255, so MAD16 < 0.02 catches 19/19 real leaks and false-flags 2 in 3000.
    """
    try:
        a = np.asarray(Image.open(p).convert("L").resize((16, 16)), dtype=float) / 255.0
        return a
    except Exception:
        return None


def near_duplicate(sig, pool):
    if sig is None:
        return False
    for other in pool:
        if other is None:
            continue
        if np.abs(sig - other).mean() < NEAR_DUP_MAD:
            return True
    return False
'''
if OLD_PSIG not in out:
    raise SystemExit("psig block not found")
out = out.replace(OLD_PSIG, NEW_PSIG)

out = out.replace("""    test_md5, test_sig = set(), set()
    for c in ["normal", "benign", "malignant"]:
        for i in (D3 / "test" / c).glob("*"):
            test_md5.add(md5(i))
            s = psig(i)
            if s:
                test_sig.add(s)""",
"""    test_md5, test_sig = set(), []
    for c in ["normal", "benign", "malignant"]:
        for i in (D3 / "test" / c).glob("*"):
            test_md5.add(md5(i))
            s = psig(i)
            if s is not None:
                test_sig.append(s)""")

out = out.replace("""                s = psig(img)
                if s and s in test_sig:
                    n_test += 1
                    continue""",
"""                if near_duplicate(psig(img), test_sig):
                    n_test += 1
                    continue""")

DST.write_text(out, encoding="utf-8")

import ast
ast.parse(out)
checks = {
    "no WILD_REPEAT constant": "WILD_REPEAT" not in out,
    "dynamic repeat": "wild_repeat = max(1" in out,
    "specB30": "specB30.pt" in out,
    "v=30": '"v": 30' in out,
    "seed 30": "random.seed(30)" in out,
    "nine added": "nine_added" in out,
    "test exclusion": "test_sig" in out,
    "MAD dedup not byte-exact": "near_duplicate(psig(img), test_sig)" in out,
    "no quantised signature": "astype(np.int8).tobytes()" not in out,
    "numpy available": "import numpy as np" in out,
}
for k, v in checks.items():
    print(f"  {'OK ' if v else 'FAIL'} {k}")
assert all(checks.values()), "transformation incomplete"
print(f"\nwrote {DST}  (syntax OK)")
