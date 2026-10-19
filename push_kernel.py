"""Push a GPU kernel to Kaggle. HARD-FAILS on Kaggle-side error instead of printing OK.

The old version printed PUSH_OK unconditionally and hid Kaggle's
"Maximum batch GPU session count of 2 reached" error for hours, which left a
kernel QUEUED forever while the caller believed it was running.
"""
import json
import os
import sys
from pathlib import Path

os.environ["KAGGLE_API_TOKEN"] = (Path.home() / ".kaggle" / "access_token").read_text().strip()

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

folder = sys.argv[1] if len(sys.argv) > 1 else "kaggle_gpu"
resp = api.kernels_push(folder)

try:
    data = json.loads(resp) if isinstance(resp, str) else resp
except Exception:
    data = {"error": f"unparseable response: {resp!r}"}

err = data.get("error")
print("ref:", data.get("ref"))
print("version:", data.get("versionNumber"))
print("kernelId:", data.get("kernelId"))
if err:
    print("KAGGLE_ERROR:", err)
    for k in ("invalidTags", "invalidDatasetSources", "invalidKernelSources"):
        if data.get(k):
            print(f"  {k}:", data[k])
    sys.exit(1)

print("PUSH_OK")
