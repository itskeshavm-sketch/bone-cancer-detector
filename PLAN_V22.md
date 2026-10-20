# V22 Plan — Lesion-aware specialist (Model 2)

**Status:** Crucible Phase 1 (planning). Spec locked in Phase 0.

## Locked spec (Phase 0 answers)

| Decision | Value |
|---|---|
| Lever | Lesion-aware training **+** higher resolution |
| GPU budget | ~5h, one aimed run |
| Success bar | Must beat **both** boards: 92.73% benchmark **and** 48.2% wild |
| Deadline | None — optimize accuracy |

**Acceptance criteria (all must hold to re-ship):**
1. Benchmark (564 films, TTA) > 92.73% **and** malignant recall ≥ 165/179
2. Wild pool (199 real films) > 48.2% at **matched** malignant recall
3. Wild malignant recall ≥ 49/95 (never trade cancer sensitivity for accuracy)
4. Verdict must hold at the *measured* operating point, not argmax

## Why this lever (evidence, not hunch)

- The specialist must currently *find* the lesion itself inside a whole radiograph.
  On a 456px view of a pelvis film, a tumor may occupy 10–20% of the pixels.
- Every capacity/schedule/loss change has been measured and rejected:
  focal (V13, 85.8%), 120M params (V14, 83.3%), MixUp+EMA (V9),
  SSL/DINO init (V19b), SWA (rejected on val in every run), longer schedule
  (V21's own curve peaks at **epoch 39**, then 21 epochs of overfitting).
- A lesion crop is a **free zoom**: it raises effective lesion resolution
  without raising the tensor size, so it carries **no new VRAM risk**.

## Architecture

```
                 ┌──────────────────────────────────────────┐
  training      │ 50% of samples: FULL image  -> 456px      │  model keeps
  per sample    │ 50% of samples: LESION CROP -> 456px      │  learning to
                 └──────────────────┬───────────────────────┘  localize AND
                                    │                        to read texture
                                    ▼
                    EfficientNet-B6  ·  456px  ·  2 classes
                    CE + label_smoothing 0.05  ·  AdamW 3e-4
                    cosine schedule  ·  45 epochs  ·  AMP
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          stageB22.pt (specialist)          stageA18.pt (gate, reused)
                    └───────────────┬───────────────┘
                                    ▼
              chained TTA (5 views) + val threshold sweep
```

**Crop is an augmentation, never a replacement.** If the model only ever saw crops
it would fail on the real demo, where nobody boxes the lesion. Mixing both views
per sample keeps localization ability while adding texture detail.

**Crop source — two sources, chosen at runtime by what the labels actually contain:**

| Label format | Crop source | Status |
|---|---|---|
| YOLO row `class cx cy w h` | true bbox, expanded by 1.6× margin, squared | probe running |
| class id only | **Grad-CAM from the reused A18 gate** (99.37% val at finding tumors) | fallback |

The kernel **auto-detects** the format at runtime, so a late probe result cannot
break the run. Both paths are implemented; the log states which one was used.

## File tree (all new unless noted)

| File | Purpose |
|---|---|
| `kaggle_gpu/train_v22.py` | the V22 kernel (training + crop + eval) |
| `kaggle_gpu/kernel-metadata.json` | **changed** → `train_v22.py` |
| `kaggle_out_probe/crop_source.py` | shared crop-source logic (bbox vs Grad-CAM) |
| `kaggle_out_probe/sweep_threshold.py` | **changed** → sweep V22's malignant threshold on wild |
| `kaggle_out_probe/eval_wild22.py` | score V22 on the 199-film wild pool |
| `checkpoints/metrics_v22.json` | results + acceptance-criteria verdict |
| `README.md` | **changed** → V22 row, new numbers if it wins |

## Dependency order

1. `crop_source.py` — pure function, no I/O. Unit-test locally on synthetic boxes.
2. `train_v22.py` — embeds crop logic, reuses the proven V21 staging/eval spine.
3. Push + poll.
4. On COMPLETE: pull weights (single-file + size check), install, sweep threshold on wild.
5. Score wild; compare against both acceptance boards.
6. Only if all four acceptance criteria hold: swap `app.py` defaults, re-run the
   end-to-end demo test, recalibrate the band table, update README.

## Contracts

- `crop_source.crop_for(img_path, label_path, mode) -> (PIL.Image, str)`
  `mode ∈ {"bbox", "cam", "none"}`; returns the original on any failure (never raises).
- `train_v22.py` writes: `specB22.pt` (best-val), `specB22_final.pt`,
  `metrics.json` (val/test TTA, per-class report, confusion), `STAGE.txt`.
- `sweep_threshold.py` reads `checkpoints/stageB22.pt`, writes
  `kaggle_out_probe/threshold_sweep_v22.json` + per-film records.

## Test strategy

| Unit | Proof |
|---|---|
| crop_source | synthetic box → crop contains the box; degenerate/empty label → passthrough, no crash |
| staging | counts match the known split (train 2,736 / val 631 / test 564) |
| training | val curve rises; best-val checkpoint saved; SWA compared on val |
| eval | confusion matrix shape 3×3; threshold sweep on val only |
| wild | matched-sensitivity comparison vs V21 (the number that decides) |
| demo | `test_uncertainty_band.py` end-to-end; served HTML carries new numbers |

## Risk list

| Risk | Mitigation |
|---|---|
| Crop is wrong (bad bbox parse) | auto-detect; crop only if a valid box parses; log the mode used; compare cropped vs uncropped val accuracy and **fall back to the uncropped run if crop hurts** |
| VRAM blow-up | crop keeps 456px → **tensor size identical to proven V21**; no new memory risk |
| Overfitting (V21 overshot at 39) | 45 epochs with cosine decay, best-val checkpointing, and SWA compared on val |
| Threshold/operating point drift | sweep malignant threshold on the wild pool at **matched** cancer recall — never trust argmax |
| Download flakiness | single-file `file_pattern` pull + hard size assertion (proven fix) |
| GPU slot cap (2 concurrent) | probe kernel is CPU-only; confirm no stray sessions before push |

## Out of scope (deliberately)

- The 3rd layer (abstain/uncertainty) — shipped already, and the user deferred it
- Gate retraining (A18 at 99.37% val is not the bottleneck)
- New data collection / web harvesting — measured exhausted
- Changing the demo's UI beyond swapping weights/thresholds
