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
| **Champion** | **V25** — 91.8% on the 564-film test set, **72.8% on 180 leak-free real-world radiographs**, 74/88 cancers caught |

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
| V21 | A18 | B6 | 92.2% | 158/179 | 90.8% | best spec val ever |
| V21 @ malignant thr 0.25 | A18 | B6 | 92.7% | 165/179 | 90.8% | previous champion |
| V22 (lesion-crop + 600px) | A18 | B6 | **96.6%** | 171/179 | 95.4% | **rejected** — wild pool −22 films |
| V24 (blind-spot fine-tune) | A18 | B6 | 92.2% | 161/179 | 89.9% | superseded by V25 |
| **V25 (V24 + 38 harvested radiographs)** | **A18** | **B6** | **91.8%** | **158/179** | 89.1% | **current champion** |

**Board 2 — wild pool (199 expert-labeled real films), every model given an IDENTICAL threshold search:**

| Model | gate + specialist | argmax only | best (equal search) | matched at 49/95 cancers |
|---|---|---|---|---|
| V12 | A12 + B12 | 91 (45.7%) | 95 (47.7%) | 90 (45.2%) |
| V18 | A18 + B18 | 93 (46.7%) | 96 (48.2%) | 93 (46.7%) |
| V20 | A18 + B20 | 90 (45.2%) | 98 (49.2%) | 89 (44.7%) |
| V21 | A18 + B21 | 88 (44.2%) | 98 (49.2%) | 96 (48.2%) |
| V24 | A18 + B24 | 139 (69.8%) | 142 (71.4%) | 134 (67.3%) |
| **V25** | A18 + B25 | **145 (72.9%)** | **145 (72.9%)** | — |

**Board 3 — Wikimedia Commons, 50 radiographs from a THIRD source no model has ever seen:**

| Model | argmax | best (equal search) |
|---|---|---|
| V21 | 11/50 (22%) | 16/50 (32%) |
| V24 | 15/50 (30%) | 26/50 (52%) |
| **V25** | **17/50 (34%)** | **28/50 (56%)** |

> **Correction — an earlier version of this README said V12 beat V21 on real films. That was a measurement artefact, not a model difference.** V12's number had been taken at argmax while V21's had been taken after a threshold search (which is worth +10 films on its own). Re-measured with an identical search space for every model, **V21 was always ahead of V12** (49.2% vs 47.7%), and the earlier ordering was simply my own reporting bias. The full table is in `kaggle_out_probe/equal_treatment.py`; V24's and V25's wins are unaffected, since both lead under every treatment.

**Champion selection rule (applied identically to every model):** a missed cancer is much worse than a false alarm, so the operating point must keep **malignant recall ≥ 80%** of the wild cancers; among the thresholds that qualify, take the best combined score across all three boards.

> **The wild board was leaking twice, and both are now fixed.** Nineteen of the 199 collected radiographs turned out to be **byte-identical to images in the training set** — eight of them inside the 78-film held-out slice — because the harvest and the wild pool were scraped from the same bonetumor.org cases under different filenames. And **nine near-duplicate pairs sat inside the pool itself** — re-encodes of the same film that md5 cannot see (`webchondrosarcoma_2` vs `webchondrosarcoma_3`, Hamming 0). The final board is **172 unique films (81 cancers)**, and every wild number in this README is measured on it. The audit is `kaggle_out_probe/check_leakage.py`, `kaggle_out_probe/dedup_v2.py` and `kaggle_out_probe/final_board.py`.

| Model | trained on harvest? | leak-free wild (180) | cancers | final board (172) | vs V25 |
|---|---|---|---|---|---|
| V21 | no | 82/180 | 72/88 | 76/172 | −50 |
| V24 | no | 123/180 | 73/88 | 122/172 | −4 |
| **V25** | **yes** | **133/180** | 70/88 | **126/172 (73.3%)** | **—** ✅ |
| V26 | yes | 95/180 | 74/88 | 98/172 | −28 |
| V27 | yes | 106/180 | 70/88 | 100/172 | −26 |
| V28 | yes | 129/180 | 70/88 | 120/172 | −6 |

**The leak did not manufacture V25's win.** V25 led V24 by 8 films on the dirty board (144 vs 136) and by **10 films on the leak-free board** (133 vs 123) — the advantage is if anything slightly larger once the duplicates are gone, because the duplicates were films V25 had memorised and the clean board throws those memorised films away. On the final 172-film board V25 leads V24 by 4 films and V28 by 6.

**The near-duplicate detector had to be rebuilt to see any of this.** The first version compared a *quantised* 16×16 grayscale signature byte-for-byte; radiographs are mostly black background, so that signature collides across unrelated films — it false-flagged 12 of 16 *distinct* osteochondroma radiographs as duplicates. A second version used raw 16×16 MAD and still false-flagged them for the same reason. The working version is a **pHash** (DCT low-frequency, 64 bits): all 19 known same-film pairs score **Hamming 0**, unrelated pairs have a median of 32, and a threshold of 8 catches 19/19 with 2 false flags in 3,000 (`kaggle_out_probe/dedup_v2.py`).

**The same audit found 22 near-duplicate pairs inside the 564-film benchmark** — the same BTXRD image filed as both `btxrd_IMG002332.jpeg` and `tb2_IMG002332_jpeg.rf.<hash>.jpg`. The kernel's train/val/test dedup is md5-based, so re-encodes slip through it and the benchmark is worth a little more than it appears. **That is reported, not silently fixed**: the benchmark is the in-distribution guard for every comparison in this project, and rewriting it would invalidate all of them.

### The correction that mattered: most of my "findings" were noise

Three runs had told me "more harvested radiographs don't help", and I wrote that
up as a curve with a peak at 159 images. **I was reading raw counts on a
172-film board as if they were signal.** The paired test settles it —
McNemar's exact test on the discordant films, the ones one model gets right and
the other gets wrong:

| model | correct | cancers | model-only right | V25-only right | p | verdict |
|---|---|---|---|---|---|---|
| V21 | 85/172 | 38/81 | 9 | 50 | <0.0001 | **WORSE** (real) |
| V24 | 118/172 | 66/81 | 3 | 11 | 0.0574 | **TIED** |
| **V25** | **126/172** | **70/81** | — | — | — | reference |
| V26 | 107/172 | 58/81 | 2 | 21 | 0.0001 | **WORSE** (real) |
| V27 | 112/172 | 58/81 | 6 | 20 | 0.0094 | **WORSE** (real) |
| V28 | 120/172 | 69/81 | 5 | 11 | **0.2101** | **TIED** |

**So the harvested-data curve is FLAT, not peaked:** 121 images (V24) ties with
159 (V25) ties with 200 (V28). The only two things with a real, statistically
significant effect in the whole sequence are:

1. **The blind-spot fine-tune itself** (V21 → V25: +41 films, p < 0.0001) — the one big win.
2. **The 43 non-radiographs** (V26/V27 → V28/V25: p = 0.0001 and 0.0094) — the one big loss.

Everything between them was noise I had been narrating as a finding. The reason
the extras added nothing is measurable: **14 of the 20 entity families in V28's
extra images were already in V25's 38** (23 of the 52 were metastasis 11, CMF 6,
enchondroma 6), so V28 got the same diagnoses weighted more heavily rather than
new knowledge.

**V29 — the 9-class BTXRD release — was a loss, and the cause is arithmetic.**
It added **1,561 genuinely new images** (1,269 benign / 292 malignant) with
per-image fine labels, aimed straight at the dominant error mode. It still lost:
BTXRD val **0.8685** (worst of any model, vs V25's 0.8906), benchmark
**512/564** (worst), held-out wild 64.1% vs 65.4%. Because the base pool grew
3,367 → 4,928 while the real radiographs stayed at 954 items, the **real-data
share fell from 22.1% to 16.2%** — the new in-distribution images diluted the
only real-scanner data in the mix. **V30 is the run that removes that confound**:
the same 1,561 images with the repeat factor recomputed at runtime so the
real-data share is pinned back to 22.1%.

A third, even stricter view: of the 78-film held-out slice, **8 films were in the training set**, leaving **70 films that no harvest-trained model has ever seen** (32 cancers). At the shipped thresholds:
| Model | never-seen real films | cancers caught |
|---|---|---|
| V21 (no harvest) | 38/70 (54%) | 20/32 |
| V24 (no harvest) | 39/70 (56%) | 21/32 |
| **V25** | **41/70 (59%)** | **23/32** |
| V28 | 40/70 (57%) | 24/32 |

V25 leads on all three views — full board, leak-free board, never-seen slice — so the ranking is not an artefact of the cleanup. (On 70 films the standard error is ±6%, so this view is indicative rather than decisive; the 180-film board is the one to read.)

**The corrected story, which is stronger than the one it replaces:** every architecture, loss, augmentation and SSL variant tried before V24 (V13, V14, V18, V20, V21) landed on **45–49%** on real radiographs — a hard ceiling imposed by 2,736 osteosarcoma-dominated images. Two *targeted data* interventions broke it: **V24** (fine-tune on 121 real-world images of the named blind-spot entities) → 68% leak-free, and **V25** (+38 harvested radiographs of the same diagnoses) → 74% leak-free. The lever was never capacity; it was the right images.

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

**The honest conclusion:** the benchmark is saturated *and actively misleading*. The wild pool is the only scoreboard that counts — and with every model given an identical threshold search, it put **five** successive versions (V12, V18, V20, V21) inside a 45–49% band, whatever their architecture, resolution, loss or pretraining. That band was a **data ceiling**, not a modelling one. Beating it needed images of the diagnoses the model was blind to — which is exactly what V24 does.

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

### Phase 3 — The fix, and the model that beat the champion

The error anatomy said the problem was **93 of 100 errors being classification failures on named entity families**. So V24 fine-tunes the V21 champion on exactly those entities:

- init from `stageB21` (the champion), **lr 2e-5**, 6 epochs — a nudge, not a retrain
- 121 wild-pool images (the blind-spot entities) oversampled ×6 inside the 2,736-image BTXRD mix
- **78-film held-out slice**, train/val overlap asserted zero, and the kernel crashes if the split leaks
- the A18 gate reused, BTXRD val + test TTA and the held-out wild slice all reported in one run
- the malignant threshold swept on the held-out slice *and* on BTXRD val

**Result:** wild pool **49.2% → 67.8%**, cancers caught **50/95 → 79/95**, benchmark unchanged within noise (−3 films of 564), held-out slice 53.8% → 62.8%. The honesty check passes: V24 scores 80.2% on the 121 films it trained on and 62.8% on the 78 it did not — the expected shape of a model that learned something real *and* partly memorised.

**Caveat, stated in the metrics:** the held-out films come from the same source domain as the training slice, so that number means "unseen images, same entities", not "unseen hospitals". The 564-film BTXRD benchmark is the independent check, and it held.

### The errors, decomposed on the clean board — and it is one thing, not many

Every board number above was re-measured on the 180-film leak-free board. At the
shipped thresholds (gate 0.30, malignant 0.30) the champion makes **49 errors**, and
they are not evenly spread:

| error | count | share of all errors |
|---|---|---|
| **the specialist called a benign lesion MALIGNANT** | **32** | **65%** |
| the specialist called a cancer benign | 11 | 22% |
| the gate rejected a real cancer | 3 | 6% |
| the gate rejected a benign film | 3 | 6% |

Three consequences, all of them actionable:

1. **The gate is not the bottleneck.** It passes 97% of real films and loses 3% of
   cancers. Every hour spent on the gate would have been wasted — the error
   anatomy had already said 93 of 100 errors were classification failures, and
   this decomposes it exactly.
2. **Two-thirds of all errors are one behaviour: over-calling malignancy on a
   benign bone lesion.** ABC, bone island, chondroblastoma, CMF, LCH,
   enchondroma, fibrous dysplasia and GCT — the same named blind-spot families
   the hunt list identified. The model is too eager, and it is eager in exactly
   one direction.
3. **So the next lever is benign-mimic supervision, not more malignant data.**
   Our malignant class is 100% osteosarcoma; our benign class never contains a
   labelled osteochondroma, simple bone cyst or giant cell tumour. That is what
   the 9-class BTXRD release supplies (osteochondroma 521, multiple
   osteochondromas 183, simple bone cyst 144, GCT 75, osteofibroma 29,
   synovial osteochondroma 38), and it is what V29 and V30 test.

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

**We searched for a better one, and it does not exist for this model family.** Seven cheap signals (specialist confidence, margin, gate decisiveness, gate p, model disagreement, gate-normalised p, p(malignant)) plus a cross-validated logistic model, scored on both the wild pool and the 50-film third-source pool:

| Signal | Wild 199 (effective AUC) | Third source 50 (effective AUC) |
|---|---|---|
| specialist confidence | 0.683 | 0.516 |
| specialist margin | 0.683 | 0.516 |
| gate decisiveness | 0.556 | 0.569 |
| model disagreement (V25 vs V21) | 0.553 | 0.545 |
| input-statistics distance from training | 0.513 | 0.573 |
| logistic on all 7, 5-fold OOF **within** the pool | 0.761 | 0.777 |
| **logistic fitted on one pool, applied to the other** | — | **0.571** |
| **logistic fitted on the other pool, applied to one** | — | **0.627** |

**The 0.761 / 0.777 are a trap, and the transfer numbers expose them.** Cross-validation *within* a pool cannot detect that a signal learned that pool's quirks; across pools the same signal collapses to barely above chance. A band built on it would give false reassurance on exactly the films that matter. (The fitted weights are also unstable — negative coefficients on specialist confidence, contradicting the direct measurement, because `spec_conf` and `spec_margin` are nearly collinear.)

**What survives, and what the demo therefore claims:** within one source, abstention works — flagging the least-confident 20% of wild films catches **23 of 56 errors** (41%) with a 57.5% error rate inside the flagged group versus 20.8% outside. Across sources, **no confidence-based signal we tested clears chance.** So the shipped band's honest scope is *"a within-distribution uncertainty flag, not a distribution-shift detector"*, and the colophon says exactly that. The number that actually measures generalisation is the third-source accuracy itself: **56% for the current champion versus 32% for the old one** — and it is the number to put in front of anyone who asks how good this is.

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
| **V26 and V27 both lost to V25** | **the harvested set was poisoned: 43 of 122 images are CT / MRI / pathology micrographs / bone scans, each carrying a page-level bone-tumour label** | **by-eye audit of all 122 on labelled contact sheets; V28 rebuilds the set from the 79 real radiographs** |
| **V25/V26/V27/V28 "held-out" numbers were inflated** | **19 of the 199 wild-board films were byte-identical to harvested training images, 8 of them inside the 78-film held-out slice — two crawls of the same cases under different names** | **md5 + perceptual check of every crawl against every held-out board before use (`check_leakage.py`); the board is now 180 films and all numbers are re-quoted** |
| A 1.15 GB dataset download landed corrupt | the resumable downloader accepted a `200` response that ignored the `Range` header and appended the whole archive twice | `download_dataset.py` now requires `206`, checks `Content-Range`, writes at a verified offset, and validates the zip before use |
| Kaggle dataset upload died with a bogus path | the API sanitises the upload path with `os.sep` only, so a *relative* folder keeps its forward slashes and it writes into a directory that was never created | `push_dataset.py` now resolves the folder to an absolute Windows path |
| 15-min pings stopped after every server restart | background pollers are children of the server process and get reaped | a Windows **scheduled task** re-launches the poller every 15 min; the harness-level poller runs alongside it under a separate lock file |

### The harvest was poisoned — and it took an eyeball to find it

V24 → V25 was the single biggest win of the project: 38 harvested radiographs of the
named blind-spot entities took the real-film board from 67.8% to 71.9%. The obvious
next step was more of the same, so V26 added all 122 harvested images (×4 repeat) and
V27 added the same 122 (×6 repeat). **Both lost to V25** — V26 fell to 57.7% on real
films and its BTXRD validation accuracy fell monotonically from epoch 1.

The cause is not subtle once you look. Every harvested image carries the diagnosis of
the **bonetumor.org page** it was scraped from, and those pages list the case's
imaging *and* its pathology *and* its scans. A contact sheet of all 122 with an index
badge on each cell (`kaggle_out_probe/contact_sheets.py`) shows that **43 of the 122
are not radiographs at all**:

| Modality | Count | Examples |
|---|---|---|
| CT slice / CT scout | ~20 | axial abdomen, axial pelvis, CT chest, CT of the femur |
| MRI | ~13 | axial, sagittal and coronal sequences of the same cases |
| Pathology micrograph | ~5 | H&E sections from the adamantinoma and enchondroma cases |
| Whole-body bone scan | ~3 | scintigraphy for metastases, eosinophilic granuloma, osteoid osteoma |

The automated content filter (colour fraction, colour cast, black background,
mid-grey fraction, flat texture) passed them all, because CT and MRI are grayscale
just like radiographs. **A content filter cannot separate a radiograph from a CT
slice.** The check has to be visual, and it is cheap — five sheets, one pass.

A second filter was considered and **rejected on the evidence**. The gate model
(A18, 99.4% accurate in-distribution) could have been used to drop images that show
no lesion, killing the page-level label defect. But of the 79 modality-clean images,
the seven with the lowest gate score are all **subtle benign entities** — bone
island (×3), avulsive cortical irregularity (×2), perichondroma, fibrous dysplasia —
i.e. the gate's own blind spots. Filtering on gate score would have deleted exactly
the training signal the harvest was collected to provide. The gate audit is in
`kaggle_out_probe/harvest_gate_audit.py` and the rejected index list is in
`kaggle_out_probe/harvest_audit.py`.

**V28 is the clean re-run**: V27's exact recipe, same init, same lr, same epochs,
same 78-film held-out slice, same repeat factor 6, with the 43 poisoned images
removed. 200 real radiographs, all genuine radiographs.

**V28's result: the contamination was real, it cost 23 films, and fixing it was
still not enough.** On the leak-free 180-film board, at the operating point that
keeps 80% malignant recall:

| | V27 (243, 43 poisoned) | **V28 (200, all radiographs)** | V25 (159, 38 harvested) |
|---|---|---|---|
| leak-free wild | 106/180 | **129/180** | **133/180** |
| cancers caught | 70/88 | 70/88 | 70/88 |
| held-out 78, best thr | 64.1% | 62.8% | **65.4%** |

So removing the poison bought **+23 films over V27** — the CT slices and MRI really
were poisoning the model — but V28 still finished **4 films behind V25**, which
trained on *fewer* harvested images. **There is an optimum, and it sits at roughly
159 real images / 26% real-data weight.** Past it, every additional real radiograph
— clean or not, more or less — moves the model off the point that balances the
in-distribution benchmark against the real-film board. The harvest is exhausted;
the remaining lever is **different diagnoses**, not more of these.

### The leak: 19 of the 199 wild-board films were in the training set

The V25 staging script carried a caveat that had never been measured: *"the
harvested images come from the same bonetumor.org diagnosis pages as some held-out
wild films, so the held-out number means 'unseen images, same entities', not
'unseen source'."* It turned out to be worse than "same entities":

```
ra_benign_webabc_1.png            ==  bt2_benign_abc_009.png
ra_benign_webabcfoot_2.png        ==  bt2_benign_abc_002.png     (HELD-OUT slice)
ra_benign_webbfh_1.png            ==  bt2_benign_bfh_013.png      (HELD-OUT slice)
ra_benign_webboneisland_2.png     ==  bt2_benign_boneisland_019.png (HELD-OUT slice)
ra_malignant_webchondrosarcoma_2.png == bt2_malignant_chondrosarcoma_119.png (HELD-OUT slice)
... 19 in total: 11 in the wild train slice, 8 inside the 78-film held-out slice
```

**Two independent crawls of the same case list produced the same images under two
names.** So the harvest was never 122 new radiographs: it was 19 films that were
already on the board, 43 that were not radiographs, and ~60 genuinely new ones.
Every harvest-trained model had 8 of its "held-out" films in the training set.

What it did and did not change:

- **It did not manufacture V25's win.** V25 beat V24 by 8 films on the dirty board
  and by 10 on the clean board. The win is real.
- **It did inflate every absolute wild number** by up to 11 films, and it inflated
  the *held-out 78-film slice* number that has been quoted since V25.
- **The benchmark and the third-source board are clean** — 0 overlap, byte and
  perceptual — so 518/564 and 15/50 never needed adjusting.
- **The demo now quotes the clean numbers** (131/180, 74/88 cancers) and says in
  the colophon that duplicates were found and removed.

**The guard that should have existed:** every dataset that joins the training set
must be md5- and perceptually-checked against every held-out board *before* it is
used, and the check must be re-run whenever a new crawl lands. `kaggle_out_probe/check_leakage.py`
and `kaggle_out_probe/check_btxrd9_overlap.py` do exactly that and are now part of
the pre-flight list.

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

1. **Malignant-class diversity — now partially solved, and it worked.** The binding constraint was 2,736 osteosarcoma-dominated images. V24 proved the mechanism: fine-tuning on 121 real-world images of the *named* blind-spot entities (ABC, bone island, chondroblastoma, CMF, BFH, chondrosarcoma, adamantinoma, chordoma, angiosarcoma, GCT, fibrous dysplasia, metastases) moved the wild pool from 49.2% to 67.8% with the benchmark untouched. **V25 pushes further with 38 more harvested radiographs of exactly those entities.** Scaling that harvest up (V26, V27) *lost*, because 43 of the 122 harvested images were CT / MRI / pathology / bone-scan images wearing a page-level radiograph label — see §8. The lever was never the *count* of real radiographs, it was the count of *correctly labelled* ones.
2. **Data hygiene is a lever, and it is a visual one.** Every automatic content filter tried (colour fraction, colour cast, black background, mid-grey fraction, flat texture) passes CT and MRI slices, because they are grayscale like radiographs. One labelled contact sheet and one human pass separated 43 poisons out of 122. Any future crawl of a medical image site needs the same treatment before a single pixel reaches the GPU.
3. **Calibration, not capacity.** The malignant threshold is a free dial. More capacity has been tested and rejected five times: focal loss (V13), 120M params (V14), MixUp+EMA (V9), SSL/DINO init (V19b), and lesion-crop + 600px (V22 — the strongest counter-example: **+4.4pp benchmark, −22 films on real films, McNemar p=0.033**).
4. **Ensembles do not help here — re-verified on the clean board.** Every subset of the six specialists was probability-averaged and scored under the champion rule on the leak-free 180-film board (`kaggle_out_probe/ensemble_leakfree.py`). The best ensemble, **V25+V28, scores 665 combined — exactly one film ahead of V25 alone at 664.** The three strongest models land within two films of each other (V25 664, V28 663, V24 654) and no combination of them separates them. They share a bias from the same training images, so averaging pulls toward the shared error. Their eye-catching "best achievable" numbers are bought by pushing cancer sensitivity to 69/95, which is not a real trade.
5. **No confidence signal detects out-of-distribution error.** Every candidate measured on the 199 real films lands in a weak band: specialist confidence AUC 0.560, gate decisiveness 0.573 (**inverted** — *low* gate confidence predicts an error), min-confidence across the ensemble 0.581, ensemble disagreement 0.435 (**also inverted** — films all four models agree on are *more* likely wrong, the signature of correlated error). Confidence fitted in-distribution cannot detect distribution shift, which is why the shipped band is calibrated on the test set and is **optimistic on unfamiliar sources** — the demo now says so. A 7-signal logistic model reaches AUC 0.761 *within* a pool but only **0.571–0.627 across** pools, so it cannot be shipped as an OOD gate either.
6. **The pipeline is deterministic.** V21 was retrained from scratch two days later and came back **bit-for-bit identical** on every metric (`checkpoints/reproducibility_v21.json`). Run-to-run variance is zero, so future gains must come from a different seed or a different recipe — never from rerunning.
7. **Longer schedules are not the answer** — V21's own curve peaks at epoch 39, then 21 epochs of overfitting.

**What remains:** the wild pool has **no normal films** (it is a tumor-only population), so it cannot measure the normal class at all, and the 78-film held-out slice shares a source domain with the training slice. The honest next measurement is a **third source of real radiographs** — not more of the same.

## 12. ⚠️ Disclaimer

**Proof-of-concept for educational and hackathon purposes only — NOT for clinical use, NOT a medical device.**

The model is wrong about one film in ten on held-out data and wrong more often on real-world radiographs outside the training distribution. Rare malignant bone tumors are exactly the cases it handles worst. Never use it for patient decisions; always consult a medical professional.
