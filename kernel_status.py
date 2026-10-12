"""Check GPU kernel status."""
import os, json
from pathlib import Path

os.environ["KAGGLE_API_TOKEN"] = (Path.home() / ".kaggle" / "access_token").read_text().strip()

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
s = api.kernels_status("juyhtrfewqasdfrgt/bone-cancer-gpu-train")
print(json.dumps(s, indent=1, default=str)[:1500])
