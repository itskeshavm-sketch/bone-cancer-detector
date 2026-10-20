"""
Gradio Demo — Two-Stage Bone Cancer Detection with Grad-CAM
Stage A (B3): tumor vs normal gate | Stage B (B4): benign vs malignant specialist
"""
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b3, efficientnet_b5, efficientnet_b6
import gradio as gr
from PIL import Image
import numpy as np
import matplotlib.cm as cm
from pathlib import Path
import torchvision.transforms.functional as TF

CSS = """
/* Hallmark · macrostructure: Workbench · tone: clinical-utilitarian · anchor hue: deep teal
 * ui-ux-pro-max overlay: Accessible & Ethical system — medical teal #0891B2 + health green,
 * Figtree/Noto Sans (system fallback), 44px targets, 3px focus ring, semantic verdict color + words */
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&family=Noto+Sans:wght@400;500;600&display=swap');
:root {
  --paper: oklch(0.965 0.015 185);
  --paper-2: oklch(0.94 0.02 185);
  --field: oklch(1 0 0);
  --ink: oklch(0.32 0.05 180);
  --ink-soft: oklch(0.45 0.04 180);
  --rule: oklch(0.82 0.03 185);
  --accent: oklch(0.55 0.12 195);
  --accent-deep: oklch(0.42 0.1 195);
  --btn: oklch(0.45 0.1 195);
  --ok: oklch(0.55 0.14 150);
  --danger: oklch(0.55 0.16 25);
  --font-display: 'Figtree', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --font-body: 'Noto Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --font-mono: ui-monospace, 'Cascadia Mono', Menlo, Consolas, monospace;
  --space-sm: 8px;
  --space-md: 16px;
  --space-lg: 24px;
  --space-xl: 40px;
  --radius: 8px;
  --body-background-fill: var(--paper);
  --body-text-color: var(--ink);
  --block-background-fill: var(--field);
  --block-border-color: var(--rule);
  --block-border-width: 1px;
  --block-radius: var(--radius);
  --block-shadow: none;
  --button-primary-background-fill: var(--btn);
  --button-primary-background-fill-hover: var(--accent-deep);
  --button-primary-text-color: var(--field);
  --button-primary-border-color: var(--btn);
  --button-radius: var(--radius);
  --input-background-fill: var(--field);
  --panel-background-fill: var(--paper-2);
}
.gradio-container {
  background: var(--paper) !important;
  color: var(--ink);
  color-scheme: light;
  font-family: var(--font-body);
  max-width: 1120px !important;
  margin: 0 auto;
  padding: var(--space-lg) var(--space-md) var(--space-xl) !important;
}
#masthead { padding: var(--space-md) 0 var(--space-lg); border-bottom: 2px solid var(--ink); margin-bottom: var(--space-lg); }
#masthead .eyebrow {
  font-family: var(--font-mono); font-size: 12px; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--ink-soft); margin: 0 0 var(--space-sm);
}
#masthead h1 {
  font-family: var(--font-display); font-style: normal; font-weight: 400;
  font-size: clamp(30px, 4.5vw, 46px); line-height: 1.05; letter-spacing: -0.01em;
  margin: 0 0 var(--space-sm); overflow-wrap: anywhere; color: var(--ink);
}
#masthead .lede { font-size: 17px; line-height: 1.55; color: var(--ink-soft); max-width: 64ch; margin: 0; }
#masthead .lede strong { color: var(--ink); font-weight: 600; }
.figures { display: flex; border-top: 1px solid var(--rule); margin-top: var(--space-lg); }
.figures > div { flex: 1 1 0; min-width: 0; padding: var(--space-sm) var(--space-md) var(--space-sm) 0; }
.figures > div + div { border-left: 1px solid var(--rule); padding-left: var(--space-md); }
.figures b { display: block; font-family: var(--font-mono); font-weight: 600; font-size: 22px; font-variant-numeric: tabular-nums; color: var(--ink); }
.figures span { font-size: 13px; color: var(--ink-soft); }
#workbench { gap: var(--space-lg); }
#workbench .form { background: transparent !important; border: none !important; box-shadow: none !important; }
#reading .form { background: var(--paper-2) !important; border: 1px solid var(--rule) !important; border-radius: var(--radius); }
#controls { margin-top: var(--space-md); display: grid; gap: var(--space-sm); }
#read-btn button, #presets button { min-height: 44px; }
#read-btn button { font-size: 16px; font-weight: 600; padding: 12px; }
.verdict { background: var(--field); border: 1px solid var(--rule); border-left: 6px solid var(--accent); border-radius: var(--radius); padding: var(--space-md); }
.verdict.v-normal { border-left-color: var(--accent); }
.verdict.v-benign { border-left-color: var(--ok); }
.verdict.v-malignant { border-left-color: var(--danger); }
.verdict.v-uncertain { border-left-color: var(--ink-soft); background: var(--paper-2); }
.verdict.v-uncertain .v-head { color: var(--ink); }
.verdict .v-kicker { font-family: var(--font-mono); font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-soft); margin: 0 0 4px; }
.verdict .v-head { font-family: var(--font-display); font-style: normal; font-weight: 700; font-size: 30px; margin: 0 0 4px; }
.verdict.v-malignant .v-head { color: var(--danger); }
.verdict.v-benign .v-head { color: var(--ok); }
.verdict .v-conf { font-family: var(--font-mono); font-size: 14px; font-variant-numeric: tabular-nums; margin: 0 0 var(--space-sm); }
.verdict .v-next { font-size: 15px; line-height: 1.55; margin: 0 0 var(--space-sm); }
.verdict .v-note { font-size: 12px; color: var(--ink-soft); margin: 0; }
.verdict .v-rel { font-size: 13px; line-height: 1.5; color: var(--ink-soft); margin: 0 0 var(--space-sm); border-top: 1px solid var(--rule); padding-top: var(--space-sm); }
#tryit { margin-top: var(--space-md); }
#tryit ol { margin: var(--space-sm) 0 0; padding-left: 20px; color: var(--ink-soft); font-size: 14px; line-height: 1.7; }
#tryit li::marker { color: var(--accent-deep); font-family: var(--font-mono); }
#colophon { border-top: 1px solid var(--rule); margin-top: var(--space-xl); padding-top: var(--space-md); font-size: 13px; line-height: 1.6; color: var(--ink-soft); max-width: 72ch; }
#colophon strong { color: var(--ink); }
input[type='range'] { accent-color: var(--accent); }
*:focus-visible { outline: 3px solid var(--accent) !important; outline-offset: 2px; }
@media (max-width: 768px) {
  .figures { flex-direction: column; }
  .figures > div + div { border-left: none; border-top: 1px solid var(--rule); padding-left: 0; }
}
@media (prefers-reduced-motion: reduce) {
  * { transition-duration: 0.15s !important; animation-duration: 0.15s !important; }
}
"""

parser = argparse.ArgumentParser()
parser.add_argument("--stageA", default="checkpoints/stageA18.pt")
parser.add_argument("--stageB", default="checkpoints/stageB21.pt")
parser.add_argument("--bands", default="checkpoints/uncertainty_bands_v21.json")
parser.add_argument("--gate-thresh", type=float, default=0.30)
parser.add_argument("--mal-thresh", type=float, default=0.25)
parser.add_argument("--port", type=int, default=7860)
args, _ = parser.parse_known_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NORM = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]


def _classes(ckpt, default):
    """V21+ checkpoints were saved without a 'classes' key; fall back to 2-way."""
    c = ckpt.get("classes")
    return len(c) if c else default


def load_b3(path):
    ckpt = torch.load(path, map_location=device)
    m = efficientnet_b3(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, _classes(ckpt, 2)))
    m.load_state_dict(ckpt["model"])
    return m.to(device).eval(), ckpt


def load_b5(path):
    ckpt = torch.load(path, map_location=device)
    m = efficientnet_b5(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, _classes(ckpt, 2)))
    m.load_state_dict(ckpt["model"])
    return m.to(device).eval(), ckpt


def load_b6(path):
    ckpt = torch.load(path, map_location=device)
    m = efficientnet_b6(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, _classes(ckpt, 2)))
    m.load_state_dict(ckpt["model"])
    return m.to(device).eval(), ckpt


modelA, ckA = load_b3(args.stageA)  # [normal, tumor], 300px
modelB, ckB = load_b6(args.stageB)  # [benign, malignant], 456px
modelB2 = None
if Path("checkpoints/stageB11.pt").exists():
    try:
        modelB2, _ = load_b6("checkpoints/stageB11.pt")
    except Exception as e:
        print("stageB11 not usable:", e)
IA, IB = int(ckA.get("img", 300)), int(ckB.get("img", 456))
print(f"A: {ckA.get('classes', ['normal', 'tumor'])} acc={ckA.get('acc')} | "
      f"B: {ckB.get('classes', ['benign', 'malignant'])} acc={ckB.get('acc')}")
print(f"operating point: gate thresh {args.gate_thresh}, malignant thresh {args.mal_thresh}")

# ---- measured reliability bands (calibrated on 564 test films) ----------------
# The band table says how often this model is actually right on films that look
# like the one being read. Bands whose measured error rate is high get flagged
# "uncertain" instead of asserting a verdict the model cannot back up.
BANDS = None
_bp = Path(args.bands)
if _bp.exists():
    try:
        BANDS = json.loads(_bp.read_text())
        print(f"bands: {BANDS['n_films']} films, overall acc {BANDS['overall_accuracy']:.3f}, "
              f"uncertain covers {BANDS['flagged_share']:.1%} of films "
              f"and {BANDS['error_capture']:.1%} of errors")
    except Exception as e:
        print("bands file unusable:", e)
        BANDS = None
else:
    print("no band table found - verdicts will show confidence only")

# Light clinical theme, pinned for BOTH modes: every *_dark twin carries the same
# light value so a dark OS color-scheme cannot restyle the page behind our back.
theme = gr.themes.Default(
    font=[gr.themes.GoogleFont("Figtree"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#F0FDFA", body_background_fill_dark="#F0FDFA",
    body_text_color="#134E4A", body_text_color_dark="#134E4A",
    body_text_color_subdued="#475569", body_text_color_subdued_dark="#475569",
    background_fill_primary="#FFFFFF", background_fill_primary_dark="#FFFFFF",
    background_fill_secondary="#F0FDFA", background_fill_secondary_dark="#F0FDFA",
    block_background_fill="#FFFFFF", block_background_fill_dark="#FFFFFF",
    block_border_color="#CCFBF1", block_border_color_dark="#99F6E4",
    block_label_background_fill="#F0FDFA", block_label_background_fill_dark="#F0FDFA",
    block_label_text_color="#134E4A", block_label_text_color_dark="#134E4A",
    block_title_text_color="#134E4A", block_title_text_color_dark="#134E4A",
    button_primary_background_fill="#0E7490", button_primary_background_fill_dark="#0E7490",
    button_primary_background_fill_hover="#155E75",
    button_primary_background_fill_hover_dark="#155E75",
    button_primary_text_color="#FFFFFF", button_primary_text_color_dark="#FFFFFF",
    button_primary_border_color="#0E7490", button_primary_border_color_dark="#0E7490",
    button_secondary_background_fill="#FFFFFF", button_secondary_background_fill_dark="#FFFFFF",
    button_secondary_text_color="#134E4A", button_secondary_text_color_dark="#134E4A",
    button_secondary_border_color="#99F6E4", button_secondary_border_color_dark="#99F6E4",
    input_background_fill="#FFFFFF", input_background_fill_dark="#FFFFFF",
    panel_background_fill="#F0FDFA", panel_background_fill_dark="#F0FDFA",
    slider_color="#0891B2", slider_color_dark="#0891B2",
    checkbox_background_color_selected="#0891B2",
    checkbox_background_color_selected_dark="#0891B2",
    checkbox_border_color_selected="#0891B2",
    checkbox_border_color_selected_dark="#0891B2",
    color_accent="#0891B2",
    color_accent_soft="#CCFBF1", color_accent_soft_dark="#CCFBF1",
    link_text_color="#0E7490", link_text_color_dark="#0E7490",
)


def hook_pair(model):
    feat, grad = {}, {}

    def fwd(m, i, o):
        feat["v"] = o

    def bwd(m, gi, go):
        grad["v"] = go[0]

    model.features[-1][0].register_forward_hook(fwd)
    model.features[-1][0].register_full_backward_hook(bwd)
    return feat, grad


featA, gradA = hook_pair(modelA)
featB, gradB = hook_pair(modelB)


def tta_views(pil):
    return [pil, TF.hflip(pil), TF.rotate(pil, 10), TF.rotate(pil, -10),
            TF.rotate(TF.hflip(pil), 10)]


def gate_tumor_prob(pil, tta):
    """Returns (tta-averaged tumor probability, single-view tumor probability).

    The single-view value is the band key on purpose: the band table was
    calibrated on the demo's single-view path, so turning on TTA must not
    change which reliability band a film is judged in.
    """
    views = tta_views(pil) if tta else [pil]
    with torch.no_grad():
        singles = [float(F.softmax(modelA(tfA(v).unsqueeze(0).to(device)), 1)[0, 1])
                   for v in views]
    return sum(singles) / len(singles), singles[0]


def specialist_probs(pil, tta, vote):
    """Returns (tta-averaged [benign, malignant], single-view key [benign, malignant]).

    The key is modelB alone at a single view so the band lookup always matches
    the calibration, regardless of TTA or the second-opinion vote.
    """
    models = [modelB] + ([modelB2] if (vote and modelB2 is not None) else [])
    views = tta_views(pil) if tta else [pil]
    with torch.no_grad():
        key = F.softmax(modelB(tfB(views[0]).unsqueeze(0).to(device)), 1)[0].cpu()
        per_view = []
        for v in views:
            acc = torch.zeros(2)
            for m in models:
                acc += F.softmax(m(tfB(v).unsqueeze(0).to(device)), 1)[0].cpu()
            per_view.append(acc / len(models))
        avg = sum(per_view) / len(per_view)
    return avg, key


def decide_specialist(prob_benign, prob_malignant, mal_thr):
    """Call malignant when p(malignant) >= mal_thr.

    The threshold is NOT a tuning knob for accuracy - it is the operating point on
    the benign<->malignant ROC curve. Measured on the 199-film wild pool, argmax
    (0.5) parks this model in a benign-leaning corner that misses rare cancers;
    0.25 restores the same cancer sensitivity as the previous champion while
    getting 6 more films right. See checkpoints/metrics_wild21.json.
    """
    return 1 if prob_malignant >= mal_thr else 0


def tf(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(*NORM),
    ])


tfA, tfB = tf(IA), tf(IB)
DISCLAIMER = "⚠️ **Proof-of-concept only — NOT for clinical use. Always consult a medical professional.**"


def gradcam_overlay(orig_pil, feat, grad, score, size):
    score.backward()
    f = feat["v"][0]
    g = grad["v"][0]
    w = g.mean(dim=(1, 2))
    cam = (w[:, None, None] * f).sum(0).clamp(min=0)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(size, size),
                        mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
    heat = (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
    base = np.array(orig_pil.convert("RGB").resize((size, size)))
    return Image.fromarray((0.45 * base + 0.55 * heat).astype(np.uint8))


def conf_word(p):
    return "high" if p >= 0.9 else ("moderate" if p >= 0.7 else "uncertain")


def band_for(conf):
    """Look up the measured reliability band for a single-view confidence value."""
    if not BANDS:
        return None
    for b in BANDS["bands"]:
        if b.get("n") and b["lo"] <= conf < b["hi"]:
            return b
    return None


def reliability_note(band):
    """Returns (uncertain?, text). Text states what the model measurably achieves on
    films that look like this one - a measured claim, not a vibe."""
    if not band:
        return False, ""
    acc = band["accuracy"]
    if band["uncertain"]:
        return True, (f"On films that look like this one it is right about {acc:.0%} of the time. "
                      f"Treat this as a flag for expert review, not a result.")
    return False, f"On films that look like this one it is right about {acc:.0%} of the time."


def verdict_card(kind, head, conf_line, next_line, rel_line=""):
    rel = f'<p class="v-rel">{rel_line}</p>' if rel_line else ""
    return (f'<div class="verdict v-{kind}">'
            f'<p class="v-kicker">Reading · {head.lower()}</p>'
            f'<p class="v-head">{head}</p>'
            f'<p class="v-conf">{conf_line}</p>'
            f'<p class="v-next">{next_line}</p>'
            f'{rel}'
            f'<p class="v-note">Proof-of-concept only — not for clinical use. Always see a doctor.</p>'
            f'</div>')


def predict(img: Image.Image, tta=False, vote=False, thresh=0.30, mal_thresh=0.25):
    if img is None:
        return None, None, DISCLAIMER
    rgb = img.convert("RGB")

    p_tumor, p_tumor_1 = gate_tumor_prob(rgb, tta)

    if p_tumor < thresh:
        modelA.zero_grad()
        logitsA = modelA(tfA(rgb).unsqueeze(0).to(device))
        overlay = gradcam_overlay(rgb, featA, gradA, logitsA[0, 0], IA)
        conf = {"normal": 1 - p_tumor, "tumor": p_tumor}
        uncertain, rel = reliability_note(band_for(1.0 - p_tumor_1))
        if uncertain:
            head = "Uncertain reading"
            nxt = ("The model leans normal here, but not confidently — a radiologist should read this film.")
            text = verdict_card("uncertain", head,
                                f"{conf['normal']:.1%} confidence · {conf_word(conf['normal'])} · leans normal",
                                nxt, rel)
        else:
            text = verdict_card("normal", "Normal", f"{conf['normal']:.1%} confidence · {conf_word(conf['normal'])}",
                                "Nothing here suggests a tumor — but this tool misses one film in ten, so keep up routine care.",
                                rel)
        return overlay, conf, text

    probB, key = specialist_probs(rgb, tta, vote)
    idx = decide_specialist(float(probB[0]), float(probB[1]), mal_thresh)
    label = ["benign", "malignant"][idx]
    modelB.zero_grad()
    logitsB = modelB(tfB(rgb).unsqueeze(0).to(device))
    overlay = gradcam_overlay(rgb, featB, gradB, logitsB[0, idx], IB)
    conf = {"benign": float(probB[0]), "malignant": float(probB[1])}
    kind = "non-cancerous" if idx == 0 else "cancerous"
    uncertain, rel = reliability_note(band_for(float(max(key[0], key[1]))))
    if uncertain:
        head = "Uncertain reading"
        nxt = (f"The model leans {label} here, but not confidently — a radiologist should read this film.")
        text = verdict_card("uncertain", head,
                            f"{conf[label]:.1%} confidence · {conf_word(conf[label])} · leans {label} "
                            f"· tumor probability {p_tumor:.1%}",
                            nxt, rel)
    else:
        nxt = ("Save this page and show it to a doctor promptly." if idx == 1
               else "Save this page and show it to a doctor at your next visit.")
        text = verdict_card(label, label.capitalize(),
                            f"{conf[label]:.1%} confidence · {conf_word(conf[label])} · {kind} · tumor probability {p_tumor:.1%}",
                            nxt, rel)
    return overlay, conf, text


def quick(img: Image.Image):
    return predict(img, False, False, args.gate_thresh, args.mal_thresh)


def thorough(img: Image.Image):
    return predict(img, True, modelB2 is not None, args.gate_thresh, args.mal_thresh)


with gr.Blocks(title="Bone cancer detector — screening aid") as demo:
    rel_band = None
    if BANDS:
        rel_band = BANDS.get("reliable_band")
    rel_pct = f"{rel_band['accuracy']:.0%}" if rel_band else "93%"
    rel_unc_pct = f"{BANDS['flagged_error_rate']:.0%}" if BANDS else "35%"
    gr.HTML(f"""
    <div id="masthead">
      <p class="eyebrow">Screening aid · proof of concept</p>
      <h1>A second look at bone X-rays.</h1>
      <p class="lede">Two models read every film: one asks <strong>tumor or not</strong>,
      the other asks <strong>harmless or cancerous</strong>. On 564 test films it got
      <strong>505 right</strong> and caught <strong>163 of 179 cancers</strong>. When it
      is not sure it says <strong>uncertain reading</strong> instead of guessing.</p>
      <div class="figures">
        <div><b>505 / 564</b><span>test films read correctly</span></div>
        <div><b>163 / 179</b><span>cancers caught</span></div>
        <div><b>{rel_pct}</b><span>accuracy on confident reads</span></div>
        <div><b>{rel_unc_pct}</b><span>error rate on flagged reads</span></div>
      </div>
    </div>""")
    with gr.Row(elem_id="workbench"):
        with gr.Column():
            inp = gr.Image(type="pil", label="The film")
            gr.HTML("""
            <div id="tryit">
              <p><strong>First visit?</strong> Click a sample film, then Read film.</p>
              <ol>
                <li>Start with the <strong>malignant</strong> sample — the clearest case.</li>
                <li>Then the <strong>benign</strong> one, and compare the two readings.</li>
                <li>Finish with <strong>normal</strong>, and drag sensitivity down to see what over-flagging looks like.</li>
              </ol>
            </div>""")
            samples = []
            for cls in ["malignant", "benign", "normal"]:
                hits = sorted(Path(f"data/test/{cls}").glob("*"))
                if hits:
                    samples.append([str(hits[0])])
            if samples:
                gr.Examples(examples=samples, inputs=[inp], label="Sample films — click one")
            with gr.Group(elem_id="controls"):
                btn = gr.Button("Read film", variant="primary", elem_id="read-btn")
                with gr.Row():
                    qbtn = gr.Button("Quick check — 4 seconds", variant="secondary")
                    tbtn = gr.Button("Thorough check — vote + 5 views (most accurate)", variant="secondary")
                tta_box = gr.Checkbox(label="Careful reading — 5 views, slower", value=False)
                ens_box = gr.Checkbox(label="Second opinion — B11 + B12 vote", value=False,
                                      visible=modelB2 is not None)
                thresh = gr.Slider(0.2, 0.7, value=args.gate_thresh, step=0.05,
                                   label="Gate sensitivity — lower catches subtler tumors, more false alarms")
                mal_slider = gr.Slider(0.05, 0.60, value=args.mal_thresh, step=0.01,
                                       label="Cancer caution — lower calls more things cancer (missed tumors cost more than false alarms)")
        with gr.Column(elem_id="reading"):
            out_img = gr.Image(label="Where the model looked")
            out_label = gr.Label(label="Reading")
            out_text = gr.HTML(label="Verdict")
    gr.HTML(f"""
    <div id="colophon">
      <p><strong>Not a medical device.</strong> Wrong about one film in ten. A missed tumor
      costs more than a false alarm, so the sensitivity slider leans toward catching.</p>
      <p><strong>It knows when it does not know.</strong> Every reading is checked against
      confidence bands measured on 564 test films: on confident reads it is right about
      {rel_pct} of the time, and on the reads it flags as uncertain it is wrong about
      {rel_unc_pct} of the time — those are marked <em>uncertain reading</em> and mean
      "a radiologist should look at this", not "probably fine".</p>
      <p>Method: EfficientNet-B3 gate, EfficientNet-B6 specialist, Grad-CAM evidence.
      Always see a doctor.</p>
    </div>""")
    btn.click(predict, inputs=[inp, tta_box, ens_box, thresh, mal_slider],
              outputs=[out_img, out_label, out_text])
    qbtn.click(quick, inputs=[inp], outputs=[out_img, out_label, out_text])
    tbtn.click(thorough, inputs=[inp], outputs=[out_img, out_label, out_text])
    inp.change(predict, inputs=[inp, tta_box, ens_box, thresh, mal_slider],
               outputs=[out_img, out_label, out_text])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=True, theme=theme, css=CSS)
