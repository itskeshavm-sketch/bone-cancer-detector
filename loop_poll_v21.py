"""Sleep 15 min, then report V21 kernel status + live GPU quota burn (one poll, then exit).

Kaggle only publishes a kernel's output files/log after the run ENDS, so live progress
is invisible. `quota_view` gives the real signal: gpuQuota.timeUsed climbs at ~1.0x wall
clock while a GPU session is actively training.
"""
import json
import time
import os
import subprocess
import sys

ROOT = r"C:\Users\Dell\OneDrive\Documents\Default Project\bone_cancer_detector"
SLUGS = ["juyhtrfewqasdfrgt/bone-cancer-v21-train"]

print("poll-sleep 900s...", flush=True)
time.sleep(900)

for slug in SLUGS:
    r = subprocess.run([sys.executable, "-c",
                        "import os;from pathlib import Path;"
                        "os.environ['KAGGLE_API_TOKEN']=(Path.home()/'.kaggle'/'access_token').read_text().strip();"
                        "from kaggle.api.kaggle_api_extended import KaggleApi;"
                        "a=KaggleApi();a.authenticate();"
                        f"s=a.kernels_status('{slug}');"
                        "print(s.status,'|',(s.failureMessage or '')[:200])"],
                       capture_output=True, text=True, cwd=ROOT)
    print(f"POLL {slug}: {(r.stdout or r.stderr).strip()[:300]}", flush=True)

q = subprocess.run([sys.executable, "-c",
                    "import os,json;from pathlib import Path;"
                    "os.environ['KAGGLE_API_TOKEN']=(Path.home()/'.kaggle'/'access_token').read_text().strip();"
                    "from kaggle.api.kaggle_api_extended import KaggleApi;"
                    "a=KaggleApi();a.authenticate();"
                    "d=json.loads(json.dumps(a.quota_view(),default=str));"
                    "d=json.loads(d) if isinstance(d,str) else d;"
                    "g=d['gpuQuota'];g=json.loads(g) if isinstance(g,str) else g;"
                    "n=lambda s: float(str(s).rstrip('s').split('.')[0]);"
                    "print(json.dumps({'used_s':n(g['timeUsed']),'reserved_s':n(g['timeReserved']),"
                    "'api_total_s':n(g['totalTimeAllowed']),'refresh':d.get('quotaRefreshTime')}))"],
                   capture_output=True, text=True, cwd=ROOT)
out = (q.stdout or "").strip()
if out:
    d = json.loads(out)
    used = d["used_s"]
    weekly_left = 108000 - used
    print(f"QUOTA gpu_used={used/3600:.2f}h  weekly_30h_left={weekly_left/3600:.2f}h "
          f"api_total={d['api_total_s']/3600:.2f}h refresh={d['refresh']}", flush=True)