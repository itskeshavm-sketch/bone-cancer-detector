"""Bounded 15-minute poller for the V26 kernel: status + live GPU quota burn.
Exits on COMPLETE/ERROR so the harness notifies.

Usage: python loop_poll_v26.py [cycles]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = r"C:\Users\Dell\OneDrive\Documents\Default Project\bone_cancer_detector"
SLUG = "juyhtrfewqasdfrgt/bone-cancer-v26-train"
cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 1

for c in range(cycles):
    print("poll-sleep 900s...", flush=True)
    time.sleep(900)
    r = subprocess.run([sys.executable, "-c",
                        "import os;from pathlib import Path;"
                        "os.environ['KAGGLE_API_TOKEN']=(Path.home()/'.kaggle'/'access_token').read_text().strip();"
                        "from kaggle.api.kaggle_api_extended import KaggleApi;"
                        "a=KaggleApi();a.authenticate();"
                        f"s=a.kernels_status('{SLUG}');"
                        "print(s.status,'|',(s.failureMessage or '')[:200])"],
                       capture_output=True, text=True, cwd=ROOT)
    status = (r.stdout or r.stderr).strip()
    print(f"POLL {c+1}: {status[:220]}", flush=True)
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
    if "COMPLETE" in status or "ERROR" in status:
        print("FINAL STATE REACHED", flush=True)
        sys.exit(0)

print("re-arm me", flush=True)
sys.exit(1)