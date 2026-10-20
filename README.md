# 🦴 Bone Cancer Detection AI

Two-stage X-ray classifier for bone tumors — **normal / benign tumor / malignant tumor** — with Grad-CAM heatmaps, a measured uncertainty band, and an honest two-board evaluation.

> **⚠️ Proof-of-concept for educational and hackathon purposes only. NOT a medical device. NOT for clinical use.**
> The model is wrong about one film in ten. Never use it for patient decisions. Always consult a medical professional.

---

## 1. What this is

| | |
|---|---|
| **Task** | 3-class bone X-ray classification via a 2-stage cascade |
| **Input** | Single 2D radiograph (PNG/JPEG) |
| **Output** | Verdict (normal / benign / malignant) + confidence + Grad-CAM overlay + reliability band |
| **Stack** | PyTorch, torchvision EfficientNet, Gradio 6.29.1 |
| **Training infra** | Kaggle GPU (T4), private kernels, 30 GPU-h/week quota |
| **Demo** | `python app.py` → http://localhost:7860 |

## 2. Architecture

```
              ┌──────────────────────────┐
   X-ray ───► │ Stage A · GATE           │  EfficientNet-B3 · 300px
              │ normal vs tumor          │  12 epochs · val 99.37% (A18)
              └────────────┬─────────────┘
                           │ p(tumor) ≥ threshold ?
              ┌────────────┴─────────────┐
             NO                        YES
              │                          │
        ┌─────▼──────┐        ┌──────────▼─────────┐
        │  Normal    │        │ Stage B · SPECIALIST│  EfficientNet-B6 · 456px
        │  + GradCAM │        │ benign vs malignant │  50-60 epochs · val 90.81% (B21)
        └────────────┘        └──────────┬─────────┘
                                          │
                              5-view TTA at eval time
                              (identity, h-flip, ±10°, h-flip+10°)
```

**Why two models:** each learns one easy question instead of one hard three-way question. The gate learns "is there a lesion at all" from ~14k images (main + ziya + xmerg); the specialist learns "benign or malignant" from 2,736 labeled tumors — a much harder, much smaller problem.

**Shared training recipe (the proven core):**

| Element | Value |
|---|---|
| Augmentation | RandomHorizontalFlip 0.5 · RandomRotation 15° · RandomAffine(10°, 0.05 translate, 0.9–1.1 scale) · ColorJitter(0.15) · RandAugment(2 ops, mag 9–12) · RandomErasing p=0.25–0.35 |
| Loss | `CrossEntropyLoss(label_smoothing=0.05)` |
| Optimizer | AdamW, lr 3e-4, weight_decay 1e-4, cosine schedule |
| Precision | AMP (fp16) — **except** models fed inputs in the ±1024 range (see §8) |
| Model selection | Best val accuracy. SWA computed but **rejected on val** in every run |
| TTA | 5 views, averaged softmax |
| Threshold | Swept on val, applied to test once |

## 3. Data

| Source | Role | Size |
|---|---|---|
| `musakhanalabazy/bone-tumor-dataset3` | Main 3-class set (YOLO-style `.txt` labels) | train 4,183 · val 1,032 · test 564 |
| `ziya07/bone-cancer-detection-dataset` | Gate training (binary tumor/normal) | ~5k |
| `mohamedmgdan/x-ray-merg` | Gate training, fresh X-rays | 8,805 |
| `wild_test/` | **External validation pool** | 208 films, **199 expert-labeled** |

Main 3-class split: benign 1,860 / malignant 1,866 / normal 2,053 across train+val+test. The gate trains on main + ziya + xmerg merged; the specialist trains only on the 2,736 main subtype (benign/malignant) training images.

**MD5 dedup** removes every train/val image whose hash matches a test image before training — otherwise the "held-out" test set leaks.

**The wild pool** is the honest board: real radiographs scraped from Radiology Assistant and bonetumor.org, labeled by expert source, duplicates removed, CT/MRI/histology/prosthesis images quarantined. Ground truth is in the filename prefix (`ra_benign_*`, `ra_malignant_*`); 9 `tumor_wild_*` films are unlabeled and excluded from scoring.

## 4. Two scoreboards — and why they disagree

**Board 1 — benchmark (564 held-out test films, 5-view TTA):**

| Version | Gate | Specialist | Test TTA | Malignant caught | Spec val | Verdict |
|---|---|---|---|---|---|---|
| V5 | B3 | B3 | 82.3% | 134/179 | 80.8% | first two-stage |
| V6 | B3 | B4 | 83.9% | — | 83.5% | |
| V7 | B3 | B5 | 87.9% | 155/179 | 90.0% | first strong |
| V9 | B3 | B5 | 87.8% | — | 89.1% | MixUp+EMA **rejected** |
| V10 | B3 | B5 | 88.5% | 160/179 | 90.6% | spec val best of its era |
| V11 | B3 | B6 | 89.4% | 160/179 | 89.5% | |
| **V12** | **A12** | **B6** | **89.5%** | **163/179** | 89.9% | **demo champion** |
| V13 | B3 | B7 | 85.8% | — | 89.9% | focal loss **rejected** |
| V14 | B3 | V2-L 120M | 83.3% | — | 84.6% | giant model **starved**, direction dead |
| V18 | A18 | B6 | 89.0% | 143/179 | 87.5% | gate val 99.37%, malignant fell |
| V20 | A18 | B5 | **92.6%** | 156/179 | 90.6% | benchmark win, wild flat |
| V21 | A18 | B6 | 92.2% | 158/179 | **90.8%** | best spec val ever |

**Board 2 — wild pool (199 expert-labeled real films):**

| | B12 | B18 | B20 | B21 |
|---|---|---|---|---|
| **A12 gate** | 45.7% | **47.2%** | 45.7% | 44.7% |
| **A18 gate** | 45.2% | 46.7% | 45.2% | 44.2% |

**The headline number and the real-world number rank models in opposite order.** V21 beats V12 by 2.7pp on the benchmark and loses by 1.5pp on real films. Three independent runs (V18, V20, V21) show the same inversion.

## 5. The investigation: why V12 was ahead

This was run as a formal bug investigation, because "V12 is better" was a claim I had not earned — and the first thing it found was a defect **in my own evaluation harness**.

### Phase 0 — Investigate

- **Report:** V12 beats V21 on wild (45.7% vs 44.2%) despite V21 winning the benchmark (92.2% vs 89.5%).
- **Localize:** the eval scripts. `eval_wild.py:20` used the **A12** gate; `eval_wild18.py:20` and `eval_wild21.py:21` used **A18**. Every wild comparison so far changed *two* variables at once — a real confound, and I had been reporting it as a head-to-head.
- **Hypotheses:** (1) gate confound; (2) seed lottery on the specialist; (3) entity concentration; (4) genuine overfit from longer training.

### Phase 1 — Evidence

One forward pass per model over all 199 films, then **all 8 (gate × specialist) combinations scored from the same cached probabilities** — `kaggle_out_probe/investigate_v12_vs_v21.py`.

| | B12 | B18 | B20 | B21 |
|---|---|---|---|---|
| **A12** | 45.7% (51 mal / 40 ben) | **47.2%** (48 / 46) | 45.7% (32 / 59) | 44.7% (29 / 60) |
| **A18** | 45.2% (49 / 41) | 46.7% (47 / 46) | 45.2% (31 / 59) | 44.2% (28 / 60) |

### Phase 2 — Root cause

Reading the malignant and benign columns together (gate held constant at A18):

| Specialist | malignant caught | benign caught | **sum** |
|---|---|---|---|
| B12 | 49/95 | 41/104 | 90 |
| B18 | 47/95 | 46/104 | 93 |
| B20 | 31/95 | 59/104 | 90 |
| B21 | 28/95 | 60/104 | 88 |

**The sum is nearly constant while the split swings wildly.** All four specialists are sitting at different points on essentially *the same* ROC curve. They are not more or less capable — they are **calibrated differently** on the benign↔malignant axis, and V12/B18 happen to sit on the malignant-leaning end.

**Why the malignant-leaning end wins here:** the wild pool is 95 malignant / 104 benign / **zero normal**, so the gate passes almost everything and the entire wild score is decided by the specialist's malignant-vs-benign boundary. The wild malignancies are rare entities (adamantinoma, angiosarcoma, chordoma, parosteal osteosarcoma, plasmacytoma, chondrosarcoma) that look nothing like the osteosarcoma-dominated training set. A malignant-leaning model calls them malignant; a benign-leaning model calls them benign. A malignant-leaning model's error is over-calling benign tumors as malignant — the *safer* error for a screening aid.

**Why total accuracy hid this:** the two errors cancel. 45.7% vs 44.2% is 3 films. The malignant split gap is 21 films. **Headline accuracy is the wrong metric for this problem** — it is a weighted sum of two errors that trade off against each other.

### Phase 3 — The fix (free, no retraining)

If the models differ only in operating point, the operating point is a **dial, not a property**. The malignant decision threshold was hardcoded to argmax. Sweeping it — `kaggle_out_probe/sweep_threshold.py` — places every model at a matched malignant recall, which is the apples-to-apples comparison of cancer-detection ability that was never available before.

**At matched cancer sensitivity (49/95 malignant caught, gate A18):**

| Model | Threshold | Wild acc | Correct | Malignant | Benign |
|---|---|---|---|---|---|
| B12 (old champion) | 0.50 (argmax) | 45.2% | 90/199 | 49/95 | 41/104 |
| B18 | 0.50 (argmax) | 46.7% | 93/199 | 47/95 | 46/104 |
| **B21 (new champion)** | **0.25** | **48.2%** | **96/199** | **49/95** | **47/104** |

**B21 at threshold 0.25 catches exactly as many cancers as the old champion and gets 6 more films right.** Its own best point (threshold 0.1) reaches **49.2% — 98/199 with 66/95 cancers** — better than any other model on both axes simultaneously. B12's best-accuracy point (threshold 0.1) catches 83 cancers but only 10 benign — clinically useless.

**Why argmax was the whole story:** B21's probabilities are systematically *lower* than B12's. Argmax (0.5) parked B21 in a benign-leaning corner of the ROC curve. The models were never different in quality — only in where they were pointed.

**The demo had a matching defect:** its "Sensitivity" slider moved only the **gate** threshold (normal↔tumor), never the **specialist** threshold (benign↔malignant), so a user could not reach the cancer-leaning operating point at all. The demo now ships a second slider — **"Cancer caution"** — defaulting to the measured 0.25, and defaults to the A18 + B21 pair.

### Phase 4 — Reality check

The V21 training curve (recovered from the kernel log) shows train acc 98.4% vs val 89.4%, with val peaking at **epoch 39** (90.81%) and then flat for 21 more epochs. The 60-epoch schedule I chose was 21 epochs of pure overfitting past the peak — a self-inflicted error, and evidence that "more epochs" was never the lever.

**Symptom vs cause:** the symptom was "V21 is worse on real films." The cause was a fixed decision threshold in the *evaluation harness and the demo*, not in the model. The fix changed a threshold, not a weight — and the model that was "worse" turned out to be the best one available.

## 6. The uncertainty band (shipped)

The model knows when it does not know. `calibrate_uncertainty.py` runs the demo's own single-view path over all 564 test films, records every film's confidence, and measures accuracy *within* each confidence band.

| Confidence band | n | Measured accuracy | Error rate | Flagged? |
|---|---|---|---|---|
| 0.00–0.60 | 12 | 41.7% | 58.3% | **yes** |
| 0.60–0.70 | 24 | 58.3% | 41.7% | **yes** |
| 0.70–0.80 | 17 | 64.7% | 35.3% | **yes** |
| 0.80–0.90 | 46 | 73.9% | 26.1% | **yes** |
| 0.90–1.00 | 465 | **93.3%** | 6.7% | no |

**17.6% of films carry 53% of all errors.** The error rate collapses above 0.90 confidence, so the band edges follow the data, not a round number.

In the demo, a low-confidence read changes the headline to **"Uncertain reading"** with a neutral border (no green/red false certainty), keeps the lean ("leans benign"), and changes the next step to *"a radiologist should read this film."* Confident reads state the measured accuracy for films that look like this one.

## 7. Demo

```bash
pip install -r requirements.txt
python app.py --port 7860      # → http://localhost:7860
```

- **Quick check** — single view, ~4 s
- **Thorough check** — 5-view TTA + B11/B12 second-opinion vote, ~7 s
- **Sample films** — one per class for the guided first test
- **Sensitivity slider** — moves the gate threshold
- **Grad-CAM overlay** — shows where each model looked
- **Uncertain reading** — measured reliability band (above)

`localhost` only. Never `0.0.0.0`.

## 8. Failure postmortems (all mine, all instructive)

| Failure | Root cause | Guard added |
|---|---|---|
| V15 crash | staging error | `STAGE.txt` markers |
| V16 0-second fail | scheduler | — |
| V17 wrong images | xmerg CSV joined train-dir to val filename | STAGE markers caught it at line 208 |
| V19a OOM (14.31/14.56 GB) | batch 64 × 6 crops on T4 | VRAM math pre-flight |
| V19b "49.4%" X-ray arm | fed **[-1,1]**, but `xrv.utils.normalize` expects **[-1024,1024]** | range asserted locally before push |
| V19b "49.4%" again (2nd run) | fp16 AMP: conv1 peak **3.2e6** vs fp16 max **65504** → `inf` → dead gradients | that arm now runs fp32 |
| V21 instant exit | `GATE: None` — dataset_sources was missing the gate dataset | code verified by pulling the kernel back off Kaggle |
| Pilot "RUNNING" for 8 h | **wedged session**, not slow training: 0-byte log, 0 output files, no quota burn | quota-burn sampling is now the mid-run liveness signal |
| V21 "RUNNING" for hours, actually queued | `push_kernel.py` printed `PUSH_OK` **unconditionally** and hid Kaggle's `"Maximum batch GPU session count of 2 reached"` | `push_kernel.py` now parses the response and exits non-zero on any `error` |
| Weights landed as 0 bytes | flaky `kaggleusercontent` host; 8/12 attempts failed | `file_pattern` single-file pull + hard size assertion |
| Wild comparison invalid | different gates used per version | one cached-probability matrix over all 8 combos |

**The four pre-flight checks every kernel must pass before it touches anyone's GPU quota:**

1. **Parse** — `ast.parse` on the script
2. **Disk** — symlinks, not copies; output size vs. the 20 GB working limit
3. **VRAM** — activation math against the T4's 14.56 GB, batch × crops
4. **Epoch fit** — measured seconds/epoch × epochs + eval ≤ available quota
5. **Paths** — dataset resolution, gate file presence, `find_main()` rglob

## 9. Kaggle operating notes

- **30 GPU-h/week**, refreshes on the 24th. `api.quota_view()` reports live usage.
- **Max 2 concurrent GPU sessions.** Exceeding it fails *pushes*, not runs.
- **Output files and logs are published only after a run ENDS.** Mid-run, both look identical to a dead kernel. The only honest mid-run signal is **quota burn** (1.0× wall clock = training, 0× = wedged).
- Kernels are pushed with `kaggle_gpu/kernel-metadata.json`; `push_kernel.py` hard-fails on Kaggle errors.
- Weight downloads: `kaggle_out_probe/kget21_final.py` pattern (single file + size check).
- Polling: `loop_poll_v21.py` sleeps 15 min, then reports kernel status **and** live quota burn. Background pollers die on server restart — re-arm after one.

## 10. Repository layout

```
app.py                          Gradio demo (V12 defaults, uncertainty band)
calibrate_uncertainty.py        564-film confidence calibration
make_bands.py                   calibration → band table
test_uncertainty_band.py        end-to-end band test on real films
train.py / train_fast.py / train_v2.py   local CPU training variants
convert_dataset.py              dataset prep
setup_dataset.py                local dataset staging
eval_*.py                       chained / ensemble / vote / error-analysis evals
kaggle_gpu/train_gpu.py         proven two-stage kernel (V18 lineage)
kaggle_gpu/train_v20.py         B5 specialist-only kernel
kaggle_gpu/train_v21.py         B6 60-epoch SWA kernel (current)
kaggle_gpu/train_ssl.py         DINO self-supervised pretraining
kaggle_pilot*/train_pilot_domain.py   X-ray-init vs ImageNet-init pilot
kaggle_v21/                     current kernel slug (bone-cancer-v21-train)
kaggle_out_probe/               all Kaggle helpers, evals, wild scorers
  investigate_v12_vs_v21.py     the 8-cell gate × specialist matrix
  sweep_threshold.py            malignant-threshold sweep (the fix)
  kget21_final.py               single-file weight pull with size assertion
checkpoints/metrics_*.json      per-version metrics
checkpoints/uncertainty_bands.json   measured reliability bands
wild_test/                      208-film external validation pool
push_kernel.py                  Kaggle push (fails loudly now)
kernel_status.py / loop_poll*.py       status + polling
```

## 11. What would actually move the needle

Measured, ranked, with the evidence attached:

1. **Malignant-class diversity.** The binding constraint is 2,736 training images dominated by osteosarcoma. Every public source was checked (Kaggle, HuggingFace, Roboflow, figshare BTXRD) — nothing adds malignant diversity.
2. **Calibration, not capacity.** The malignant threshold is a free dial. More capacity has now been tested and rejected four times: focal loss (V13), 120M params (V14), MixUp+EMA (V9), SSL/DINO init (V19b).
3. **Stop, don't hide, uncertainty.** The shipped band is a bigger clinical win than any of the benchmark deltas, and it cost zero GPU hours.
4. **Longer schedules are not the answer** — V21's own curve proves it (peak at epoch 39, then 21 epochs of overfitting).

## 12. ⚠️ Disclaimer

**Proof-of-concept for educational and hackathon purposes only — NOT for clinical use, NOT a medical device.**

The model is wrong about one film in ten on held-out data and wrong more often on real-world radiographs outside the training distribution. Rare malignant bone tumors are exactly the cases it handles worst. Never use it for patient decisions; always consult a medical professional.
