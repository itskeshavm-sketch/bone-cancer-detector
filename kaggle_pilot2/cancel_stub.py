"""Terminate stub: pushed to cancel a wedged pilot session and free the GPU slot for V21.
The real pilot code lives in git (kaggle_pilot2/train_pilot_domain.py) and can be
re-pushed any time. This kernel exists only to occupy the slug safely."""
import sys
print("PILOT CANCELLED BY DESIGN - freeing GPU slot for V21", flush=True)
sys.exit(0)
