"""Sleep 15 min, then report Kaggle kernel status (one poll, then exit)."""
import time
import os
import subprocess
import sys

print("poll-sleep 900s...", flush=True)
time.sleep(900)
for attempt in range(1, 6):
    r = subprocess.run([sys.executable, "kernel_status.py"], capture_output=True, text=True,
                       cwd=r"C:\Users\Dell\OneDrive\Documents\Default Project\bone_cancer_detector")
    if r.returncode == 0 and "status" in (r.stdout or ""):
        break
    print(f"poll attempt {attempt} failed, retrying...", flush=True)
    time.sleep(60 * attempt)
print("POLL-RESULT:", (r.stdout or "").strip()[:500], flush=True)
if r.stderr:
    print("POLL-STDERR:", r.stderr.strip()[-300:], flush=True)
