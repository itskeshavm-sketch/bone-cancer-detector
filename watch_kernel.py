"""Wait for GPU kernel, then download outputs."""
import os, time, json
from pathlib import Path

os.environ["KAGGLE_API_TOKEN"] = (Path.home() / ".kaggle" / "access_token").read_text().strip()

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

SLUG = "juyhtrfewqasdfrgt/bone-cancer-gpu-train"
for i in range(120):  # up to 4h
    s = api.kernels_status(SLUG)
    st = s.get("status") if isinstance(s, dict) else str(s)
    print(f"[{i}] {st}", flush=True)
    if st in ("COMPLETE", "FAILED", "CANCELLED"):
        print(json.dumps(s, default=str)[:800], flush=True)
        break
    time.sleep(120)

api.kernels_output(SLUG, path="kaggle_out", force=True)
print("OUTPUT_DOWNLOADED", flush=True)
for f in sorted(Path("kaggle_out").rglob("*")):
    print(" -", f, f.stat().st_size if f.is_file() else "", flush=True)
