"""Single-cycle poller for the V22 kernel: sleep 15 min, report, EXIT.

Exiting after ONE cycle is deliberate. The harness notifies me when a background
job COMPLETES, so a multi-cycle poller inside one process would silence me for
the whole hour it covers. One sleep + one report + one exit = one ping every
15 minutes, which is what a ping-per-cycle is supposed to mean.

Usage: python loop_poll_v22.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = r"C:\Users\Dell\OneDrive\Documents\Default Project\bone_cancer_detector"
SLUG = "juyhtrfewqasdfrgt/bone-cancer-v22-train"

print("poll-sleep 900s...", flush=True)
time.sleep(900)

r = subprocess.run([sys.executable, "-c",
                    "import os;from pathlib import Path;"
                    "os.environ['KAGGLE_API_TOKEN']=(Path.home()/'.kaggle'/'access_token').read_text().strip();"
                    "from kaggle.api.kaggle_api_extended import KaggleApi;"
                    "a=KaggleApi();a.authenticate();"
                    f"s=a.kernels_status('{SLUG}');"
                    "print(s.status,'|',(s.failureMessage or '')[:300])"],
                   capture_output=True, text=True, cwd=ROOT)
print("POLL:", (r.stdout or r.stderr).strip()[:400], flush=True)

q = subprocess.run([sys.executable, "-c",
                    "import os,json;from pathlib import Path;"
                    "os.environ['KAGGLE_API_TOKEN']=(Path.home()/'.kaggle'/'access_token').read_text().strip();"
                    "from kaggle.api.kaggle_api_extended import KaggleApi;"
                    "a=KaggleApi();a.authenticate();"
                    "d=json.loads(a.quota_view().to_json());"
                    "g=d['gpuQuota'];g=json.loads(g) if isinstance(g,str) else g;"
                    "n=lambda s: float(str(s).rstrip('s').split('.')[0]);"
                    "print(json.dumps({'used':n(g['timeUsed'])}))"],
                   capture_output=True, text=True, cwd=ROOT)
if q.stdout.strip():
    used = json.loads(q.stdout.strip())["used"]
    print(f"QUOTA gpu_used={used/3600:.2f}h weekly_left={(108000-used)/3600:.2f}h", flush=True)