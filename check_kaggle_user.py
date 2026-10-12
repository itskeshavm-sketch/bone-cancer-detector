"""Check Kaggle auth and discover account username for kernel push."""
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

tok = (Path.home() / ".kaggle" / "access_token").read_text().strip()
import os
os.environ["KAGGLE_API_TOKEN"] = tok

api = KaggleApi()
api.authenticate()
print("AUTH_OK")

for key in ("username", "user"):
    try:
        print(f"config[{key}] =", api.get_config_value(key))
    except Exception as e:
        print(f"config[{key}] fail:", str(e)[:120])

try:
    ks = api.kernels_list(mine=True, page_size=3)
    print("MINE:", ks)
except Exception as e:
    print("kernels_list(mine) fail:", str(e)[:300])

try:
    ds = api.dataset_list(mine=True, max_size=3)
    print("DATASETS_MINE:", [(d.ref, getattr(d, "creatorName", None)) for d in (ds or [])])
except Exception as e:
    print("dataset_list(mine) fail:", str(e)[:300])
