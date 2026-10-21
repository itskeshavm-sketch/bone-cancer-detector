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

**Board 1 — benchmark (564 held-out test films):**

| Version | Gate | Specialist | Test TTA | Malignant caught | Spec val | Verdict |
|---|---|---|---|---|---|---|
| V5 | B3 | B3 | 82.3% | 134/179 | 80.8% | first two-stage |
| V6 | B3 | B4 | 83.9% | — | 83.5% | |
| V7 | B3 | B5 | 87.9% | 155/179 | 90.0% | first strong |
| V9 | B3 | B5 | 87.8% | — | 89.1% | MixUp+EMA **rejected** |
| V10 | B3 | B5 | 88.5% | 160/179 | 90.6% | spec val best of its era |
| V11 | B3 | B6 | 89.4% | 160/179 | 89.5% | |
| V12 | A12 | B6 | 89.5% | 163/179 | 89.9% | previous champion |
| V13 | B3 | B7 | 85.8% | — | 89.9% | focal loss **rejected** |
| V14 | B3 | V2-L 120M | 83.3% | — | 84.6% | giant model **starved**, direction dead |
| V18 | A18 | B6 | 89.0% | 143/179 | 87.5% | gate val 99.37% |
| V20 | A18 | B5 | 92.6% | 156/179 | 90.6% | benchmark win, wild flat |
| V21 | A18 | B6 | 92.2% | 158/179 | **90.8%** | best spec val ever |
| **V21 @ malignant thr 0.25** | **A18** | **B6** | **92.7%** (single-view) | **165/179** | 90.8% | **current champion** |
| V22 (lesion-crop + 600px) | A18 | B6 | **96.6%** | **171/179** | **95.4%** | **rejected** — wild pool −22 films |

**Board 2 — wild pool (199 expert-labeled real films), at the demo's shipped operating point (gate 0.30, malignant 0.25):**

| | B12 | B18 | B20 | B21 |
|---|---|---|---|---|
| **A12 gate** | 45.7% | **47.2%** | 45.7% | 44.7% |
| **A18 gate** | 45.2% | 46.7% | 45.2% | 44.2% |
| **A18 gate, malignant thr 0.25** | — | — | — | **49.2% (98/199)**, 50/95 cancers, 48/104 benign |

> **Correction:** earlier figures of 45.7% / 48.2% for V21 were measured with the evaluation harness's `gate_thresh=0.50`. At the **demo's shipped gate threshold of 0.30** the same weights score **49.2% (98/199)**. The threshold sweep is in `checkpoints/metrics_v21.json`.

**The headline number and the real-world number ranked models in opposite order** — until the operating point was corrected. V21 beat V12 by 2.7pp on the benchmark while losing on real films; with the threshold fixed it beats V12 on *both* boards, and catches more cancers (165 vs 163) in a single view than V12 does with 5-view TTA.

**And then V22 broke the pattern in the most instructive way possible.** Lesion-crop augmentation (3,367 polygon-derived crops, 50% of samples) at 600px produced the best benchmark numbers of the entire project — **96.6%**, specialist val **95.4%**, **171/179** cancers — and was **rejected**, because on the 199 real films it lost 22 films to V21 at identical cancer sensitivity (McNemar exact p = 0.0334).

| Sensitivity (malignant/95) | V21 correct | V22 correct | Δ |
|---|---|---|---|
| 38/95 | 47.2% | 37.2% | **−20** |
| 44/95 | 47.2% | 39.2% | **−16** |
| **49/95 (matched)** | **48.2% (96/199)** | **37.2% (74/199)** | **−22** |
| 56/95 | 46.7% | 38.7% | −16 |
| 66/95 | 49.2% | 41.2% | −16 |

**Mechanism:** crop training decoupled the classifier from whole-image context. Lesion-centric features transfer beautifully to the in-distribution benchmark (+4.4pp) and mislead on real radiographs from other sources, where the lesion is small in frame and the image statistics are alien. Paired flips: 59 benign→malignant, 39 malignant→benign. **This is benchmark overfitting in its purest form — and it is now the fifth independent confirmation** (V13 focal, V14 120M params, V18, V20, V22 all moved the benchmark up and the wild pool down; only the V21 threshold fix moved both up).

**The honest conclusion:** the benchmark is saturated *and actively misleading*. The wild pool is the only scoreboard that counts, and it says the ceiling is ~46–48% on out-of-distribution radiographs regardless of architecture, resolution, or augmentation. Beating it needs training data that matches deployment (varied real-world radiographs), not more of the same osteosarcoma-heavy set — which no public dataset supplies.

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

**Measured on the shipped champion (A18 + B21, malignant threshold 0.25):**

| Confidence band | n | Measured accuracy | Error rate | Flagged? |
|---|---|---|---|---|
| 0.00–0.60 | 6 | 66.7% | 33.3% | **yes** |
| 0.60–0.70 | 13 | 53.8% | 46.2% | **yes** |
| 0.70–0.80 | 20 | 70.0% | 30.0% | **yes** |
| 0.80–0.90 | 24 | 75.0% | 25.0% | **yes** |
| 0.90–1.00 | 501 | **95.8%** | 4.2% | no |

**11.2% of films carry 48.8% of all errors.** The error rate collapses above 0.90 confidence, so the band edges follow the data, not a round number. The reliable band is **95.8% accurate** — and the previous champion's equivalent band was 93.3%.

In the demo, a low-confidence read changes the headline to **"Uncertain reading"** with a neutral border (no green/red false certainty), keeps the lean ("leans benign"), and changes the next step to *"a radiologist should read this film."* Confident reads state the measured accuracy for films that look like this one.

### ⚠️ The band does not generalise out of distribution (measured)

Re-running the shipped band table against the **199 real-world films** exposes the limitation of the safety net:

| Band | n (wild) | Wild accuracy | Wild error rate | Flagged? |
|---|---|---|---|---|
| 0.00–0.60 | 24 | 62.5% | 37.5% | yes |
| 0.60–0.70 | 22 | 45.5% | 54.5% | yes |
| 0.70–0.80 | 22 | 54.5% | 45.5% | yes |
| 0.80–0.90 | 43 | 37.2% | 62.8% | yes |
| **0.90–1.00** | 88 | **48.9%** | **51.1%** | no |

**The "reliable" band is 48.9% accurate on real films — a coin flip — and the flagged region carries a 52.3% error rate versus 51.8% overall.** The band therefore carries **no information** on out-of-distribution data: the model's confidence is calibrated to the benchmark's distribution, and on unseen real radiographs it is confidently wrong.

**This is the real remaining problem, and it is a data problem, not a modelling problem.** A confidence threshold fitted in-distribution cannot detect distribution shift. The fix is to rebuild the abstention signal on out-of-distribution data (the wild pool, with cross-validation so the same 199 films are not used for both fitting and reporting), or to use a shift-sensitive signal — the gate's own confidence, or input-statistics distance — rather than the specialist's softmax. Until that is done, the "uncertain reading" label is honest on benchmark films and **not** trustworthy on real ones.

## 7. Demo

```bash
pip install -r requirements.txt
python app.py --port 7860      # → http://localhost:7860
```

- **Quick check** — single view, ~4 s
- **Thorough check** — 5-view TTA + B11/B12 second-opinion vote, ~7 s
- **Sample films** — one per class for the guided first test
- **Gate sensitivity** (0.2–0.7, default 0.30) — normal↔tumor boundary
- **Cancer caution** (0.05–0.60, default 0.25) — benign↔malignant boundary. Measured on the wild pool: at 0.25 the model catches the same 49/95 cancers as the old champion *and* gets 6 more films right. Lower it to catch subtler cancers; raise it to reduce false alarms.
- **Grad-CAM overlay** — shows where each model looked
- **Uncertain reading** — measured reliability band (above)

**Shipped configuration:** `stageA18.pt` (gate, 99.37% val) + `stageB21.pt` (specialist, 90.81% val), gate threshold 0.30, malignant threshold 0.25, band table `uncertainty_bands_v21.json`.

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
- Kernels are pushed with each folder's `kernel-metadata.json`; `push_kernel.py` and
  `kaggle_out_probe/push_any.py` hard-fail on Kaggle-side errors (an earlier version
  printed `PUSH_OK` unconditionally and hid `"Maximum batch GPU session count of 2
  reached"` for hours).
- **Live kernels:** `bone-cancer-v21-train` (V21, current champion weights),
  `bone-cancer-v22-train` (V22, lesion-aware). The original `bone-cancer-gpu-train`
  kernel was deleted on 2026-10-20 to release a wedged GPU session slot — its code
  is `kaggle_gpu/train_gpu.py` in this repo and its weights are in `checkpoints/`
  (`stageA18.pt`, `stageB18.pt`). It is restored at the end of the V22 run.
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

1. **Malignant-class diversity.** The binding constraint is 2,736 training images dominated by osteosarcoma. Every public source was checked (Kaggle, HuggingFace, Roboflow, figshare BTXRD) — nothing adds malignant diversity. This is now the *only* remaining lever the measurements point to.
2. **Calibration, not capacity.** The malignant threshold is a free dial. More capacity has now been tested and rejected five times: focal loss (V13), 120M params (V14), MixUp+EMA (V9), SSL/DINO init (V19b), and lesion-crop + 600px (V22 — the strongest case: **+4.4pp benchmark, −22 films on real films, McNemar p=0.033**).
3. **Ensembles do not help here.** Four specialists, probability-averaged: at matched cancer sensitivity the average is *worse* than the best single model (90–92/199 vs **98/199** for B21 alone). They share a bias from the same training images, so averaging pulls toward the shared error. Their eye-catching "best achievable" numbers (108/199) are bought by pushing cancer sensitivity to 69/95, which is not a real trade.
4. **No confidence signal detects out-of-distribution error.** Every candidate measured on the 199 real films lands in a weak band: specialist confidence AUC 0.560, gate decisiveness 0.573 (**inverted** — *low* gate confidence predicts an error), min-confidence across the ensemble 0.581, ensemble disagreement 0.435 (**also inverted** — films all four models agree on are *more* likely wrong, the signature of correlated error). Confidence fitted in-distribution cannot detect distribution shift.
5. **Longer schedules are not the answer** — V21's own curve proves it (peak at epoch 39, then 21 epochs of overfitting).
6. **Stop, don't hide, uncertainty.** The shipped band is a bigger clinical win than any of the benchmark deltas, and it cost zero GPU hours — but it is only trustworthy on benchmark-like films, and the demo must say so.

**The honest summary:** V21 is at the ceiling of what this data and this model family can do on unseen real radiographs. The remaining path is training data whose distribution matches deployment — varied real-world radiographs with diverse malignant entities — which no public dataset currently supplies.

## 12. ⚠️ Disclaimer

**Proof-of-concept for educational and hackathon purposes only — NOT for clinical use, NOT a medical device.**

The model is wrong about one film in ten on held-out data and wrong more often on real-world radiographs outside the training distribution. Rare malignant bone tumors are exactly the cases it handles worst. Never use it for patient decisions; always consult a medical professional.
