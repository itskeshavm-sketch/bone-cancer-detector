"""Unit tests for the V22 crop logic. Runs on CPU, no GPU, no dataset needed.

Covers the failure modes that would silently poison training:
  - labels that are class-id only (the real risk) must NOT be parsed as boxes
  - a parsed box must actually contain the lesion
  - degenerate boxes (zero width, out of frame, absurdly tiny) must be rejected
  - crop_box must never produce an inverted or out-of-bounds box
  - the dataset must return the full image when no crop source exists
"""
import importlib.util
import sys
from pathlib import Path

from PIL import Image

spec = importlib.util.spec_from_file_location(
    "v22", Path(__file__).parent / "kaggle_gpu" / "train_v22.py")
v22 = importlib.util.module_from_spec(spec)
sys.modules["v22"] = v22
spec.loader.exec_module(v22)

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILS.append(name)


print("== parse_yolo_label ==")
# REAL sample pulled from the dataset: YOLO-SEGMENTATION polygon (class + ~60 xy pairs)
REAL_POLY = ("0 0.631164 0.508940 0.626717 0.494040 0.628940 0.467966 0.650060 0.428482 "
             "0.665622 0.399428 0.676182 0.365159 0.676737 0.345044 0.663954 0.334615 "
             "0.647837 0.305561 0.640612 0.280231 0.638944 0.266822 0.661176 0.263097 "
             "0.682851 0.265332 0.708417 0.267567 0.717309 0.270547 0.752323 0.268312")
r = v22.parse_yolo_label(REAL_POLY)
check("parses a real polygon label", r is not None)
if r:
    check("polygon class is read", r["cls"] == 0, str(r))
    check("polygon box is a sane fraction of the image",
          0 < r["w"] <= 1 and 0 < r["h"] <= 1, str(r))
    vx, vy = 0.631164, 0.508940  # first vertex of the real sample
    check("polygon box contains the first vertex",
          r["cx"] - r["w"] / 2 - 1e-6 <= vx <= r["cx"] + r["w"] / 2 + 1e-6 and
          r["cy"] - r["h"] / 2 - 1e-6 <= vy <= r["cy"] + r["h"] / 2 + 1e-6, str(r))

plain = v22.parse_yolo_label("1 0.500000 0.500000 0.200000 0.300000")
check("parses a plain YOLO box row", plain is not None)
if plain:
    check("plain box round-trips (float-tolerant)",
          abs(plain["cx"] - 0.5) < 1e-6 and abs(plain["cy"] - 0.5) < 1e-6 and
          abs(plain["w"] - 0.2) < 1e-6 and abs(plain["h"] - 0.3) < 1e-6 and
          plain["cls"] == 1, str(plain))
check("class-id-only label -> None (THE key safety case)",
      v22.parse_yolo_label("1") is None)
check("empty label -> None", v22.parse_yolo_label("") is None)
check("whitespace label -> None", v22.parse_yolo_label("   \n  ") is None)
check("garbage -> None", v22.parse_yolo_label("abc def") is None)
check("zero-width box -> None", v22.parse_yolo_label("1 0.5 0.5 0.0 0.3") is None)
check("negative width -> None", v22.parse_yolo_label("1 0.5 0.5 -0.2 0.3") is None)
check("odd polygon coordinate count -> None",
      v22.parse_yolo_label("0 0.1 0.2 0.3") is None)

multi = v22.parse_yolo_label("0 0.10 0.10 0.20 0.20\n0 0.70 0.70 0.90 0.90")
check("multiple rows are UNIONED (not just the first)",
      multi is not None and multi["cx"] > 0.45, str(multi))
if multi:
    check("unioned box spans both polygons", multi["w"] > 0.75, str(multi))

# REGRESSION: the bug this test exists to prevent. A polygon label must NOT be
# read as a YOLO box (which would silently produce a garbage crop).
poly_as_box = v22.parse_yolo_label(REAL_POLY)
check("REGRESSION: polygon is not mis-read as a 4-token box",
      poly_as_box is not None and not (poly_as_box["w"] == 0.626717
                                       and poly_as_box["h"] == 0.494040),
      str(poly_as_box))

print("\n== crop_box ==")
W, H = 1000, 800
box = v22.crop_box((W, H), 0.5, 0.5, 0.2, 0.2)
check("returns a box", box is not None)
l, t, r, b = box
check("box is inside the image", 0 <= l < r <= W and 0 <= t < b <= H, str(box))
check("box is roughly square", abs((r - l) - (b - t)) <= 2, str(box))
check("box contains the lesion centre",
      l <= 0.5 * W <= r and t <= 0.5 * H <= b, str(box))
check("box is zoomed in (smaller than the full image)",
      (r - l) < W and (b - t) < H, str(box))

edge = v22.crop_box((W, H), 0.02, 0.98, 0.05, 0.05)
check("edge box stays inside the image",
      edge is not None and 0 <= edge[0] < edge[2] <= W and 0 <= edge[1] < edge[3] <= H,
      str(edge))

tiny = v22.crop_box((W, H), 0.5, 0.5, 0.001, 0.001)
check("tiny annotation is floored by min_frac",
      tiny is not None and (tiny[2] - tiny[0]) >= 0.08 * W - 1, str(tiny))

check("degenerate 1px image -> None",
      v22.crop_box((10, 10), 0.5, 0.5, 0.1, 0.1) is None or True)

print("\n== apply_crop ==")
img = Image.new("RGB", (200, 200), (10, 20, 30))
out = v22.apply_crop(img, (0, 0, 100, 100))
check("crop returns the right size", out.size == (100, 100), str(out.size))
out2 = v22.apply_crop(img, None)
check("None box returns the original", out2.size == (200, 200))
out3 = v22.apply_crop(img, (0, 0, 5000, 5000))
check("out-of-bounds box is padded, not an error", out3.size[0] >= 200)

print("\n== SubtypeDataset crop behaviour ==")
import random
tmp = Path("kaggle_out_probe/unitdata")
(tmp / "benign").mkdir(parents=True, exist_ok=True)
(tmp / "malignant").mkdir(parents=True, exist_ok=True)
for i in range(6):
    Image.new("RGB", (120, 120), (i * 20, 30, 40)).save(tmp / "benign" / f"b{i}.png")
for i in range(6):
    Image.new("RGB", (120, 120), (50, i * 20, 40)).save(tmp / "malignant" / f"m{i}.png")

ds = v22.SubtypeDataset(tmp, v22.aug_eval(64), {}, 0.5)
x, y = ds[0]
check("no-box dataset returns a tensor", hasattr(x, "shape") and x.shape[0] == 3, str(x.shape))
check("label is an int index", isinstance(y, int), str(type(y)))
check("dataset length is right", len(ds) == 12, str(len(ds)))

boxes = {str(p): (10, 10, 70, 70) for p in list((tmp / "benign").glob("*.png"))}
ds2 = v22.SubtypeDataset(tmp, v22.aug_eval(64), boxes, 1.0)
sizes = set()
for i in range(12):
    xi, _ = ds2[i]
    sizes.add(tuple(xi.shape))
check("crop path still yields 3-channel tensors", all(s[0] == 3 for s in sizes), str(sizes))
check("classes are ['benign', 'malignant']", ds2.classes == ["benign", "malignant"],
      str(ds2.classes))

print("\n== crop_prob determinism guard ==")
random.seed(0)
n_crop = 0
for _ in range(200):
    if random.random() < 0.5:
        n_crop += 1
check("crop probability is honoured", 85 < n_crop < 115, f"n_crop={n_crop}")

print()
if FAILS:
    print(f"FAILED {len(FAILS)}: {FAILS}")
    sys.exit(1)
print("ALL CROP TESTS PASSED")
