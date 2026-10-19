"""Sleep 15 min, then report BOTH pilot kernels' status (one poll, then exit)."""
import time
import os
import subprocess
import sys

ROOT = r"C:\Users\Dell\OneDrive\Documents\Default Project\bone_cancer_detector"
SLUGS = ["juyhtrfewqasdfrgt/bone-cancer-pilot-domain-init"]

print("poll-sleep 900s...", flush=True)
time.sleep(900)

for slug in SLUGS:
    r = subprocess.run([sys.executable, "-c",
                        "import os;from pathlib import Path;"
                        "os.environ['KAGGLE_API_TOKEN']=(Path.home()/'.kaggle'/'access_token').read_text().strip();"
                        "from kaggle.api.kaggle_api_extended import KaggleApi;"
                        "a=KaggleApi();a.authenticate();"
                        f"s=a.kernels_status('{slug}');"
                        "print(s.status, '|', (s.failureMessage or '')[:200])"],
                       capture_output=True, text=True, cwd=ROOT)
    print(f"POLL {slug}: {(r.stdout or r.stderr).strip()[:300]}", flush=True)