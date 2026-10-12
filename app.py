"""
Gradio Demo — Two-Stage Bone Cancer Detection with Grad-CAM
Stage A (B3): tumor vs normal gate | Stage B (B4): benign vs malignant specialist
"""
import argparse
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
  margin: 0 0 var(--space-sm); overflow-wrap: anywhere;
}
#masthead .lede { font-size: 17px; line-height: 1.55; color: var(--ink-soft); max-width: 64ch; margin: 0; }
#masthead .lede strong { color: var(--ink); font-weight: 600; }
.figures { display: flex; border-top: 1px solid var(--rule); margin-top: var(--space-lg); }
.figures > div { flex: 1 1 0; min-width: 0; padding: var(--space-sm) var(--space-md) var(--space-sm) 0; }
.figures > div + div { border-left: 1px solid var(--rule); padding-left: var(--space-md); }
.figures b { display: block; font-family: var(--font-mono); font-weight: 600; font-size: 22px; font-variant-numeric: tabular-nums; }
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
.verdict .v-kicker { font-family: var(--font-mono); font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-soft); margin: 0 0 4px; }
.verdict .v-head { font-family: var(--font-display); font-style: normal; font-weight: 700; font-size: 30px; margin: 0 0 4px; }
.verdict.v-malignant .v-head { color: var(--danger); }
.verdict.v-benign .v-head { color: var(--ok); }
.verdict .v-conf { font-family: var(--font-mono); font-size: 14px; font-variant-numeric: tabular-nums; margin: 0 0 var(--space-sm); }
.verdict .v-next { font-size: 15px; line-height: 1.55; margin: 0 0 var(--space-sm); }
.verdict .v-note { font-size: 12px; color: var(--ink-soft); margin: 0; }
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
parser.add_argument("--stageA", default="checkpoints/stageA12.pt")
parser.add_argument("--stageB", default="checkpoints/stageB12.pt")
parser.add_argument("--port", type=int, default=7860)
args, _ = parser.parse_known_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NORM = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]


def load_b3(path):
    ckpt = torch.load(path, map_location=device)
    m = efficientnet_b3(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, len(ckpt["classes"])))
    m.load_state_dict(ckpt["model"])
    return m.to(device).eval(), ckpt


def load_b5(path):
    ckpt = torch.load(path, map_location=device)
    m = efficientnet_b5(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, len(ckpt["classes"])))
    m.load_state_dict(ckpt["model"])
    return m.to(device).eval(), ckpt


def load_b6(path):
    ckpt = torch.load(path, map_location=device)
    m = efficientnet_b6(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, len(ckpt["classes"])))
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
print(f"A: {ckA['classes']} acc={ckA.get('acc')} | B: {ckB['classes']} acc={ckB.get('acc')}")


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
    views = tta_views(pil) if tta else [pil]
    with torch.no_grad():
        tp = sum(float(F.softmax(modelA(tfA(v).unsqueeze(0).to(device)), 1)[0, 1])
                 for v in views) / len(views)
    return tp


def specialist_probs(pil, tta, vote):
    models = [modelB] + ([modelB2] if (vote and modelB2 is not None) else [])
    views = tta_views(pil) if tta else [pil]
    with torch.no_grad():
        p = sum(F.softmax(m(tfB(v).unsqueeze(0).to(device)), 1)[0].cpu()
                for m in models for v in views)
    return p / (len(models) * len(views))


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


def verdict_card(kind, head, conf_line, next_line):
    return (f'<div class="verdict v-{kind}">'
            f'<p class="v-kicker">Reading · {head.lower()}</p>'
            f'<p class="v-head">{head}</p>'
            f'<p class="v-conf">{conf_line}</p>'
            f'<p class="v-next">{next_line}</p>'
            f'<p class="v-note">Proof-of-concept only — not for clinical use. Always see a doctor.</p>'
            f'</div>')


def predict(img: Image.Image, tta=False, vote=False, thresh=0.5):
    if img is None:
        return None, None, DISCLAIMER
    rgb = img.convert("RGB")

    p_tumor = gate_tumor_prob(rgb, tta)

    if p_tumor < thresh:
        modelA.zero_grad()
        logitsA = modelA(tfA(rgb).unsqueeze(0).to(device))
        overlay = gradcam_overlay(rgb, featA, gradA, logitsA[0, 0], IA)
        conf = {"normal": 1 - p_tumor, "tumor": p_tumor}
        text = verdict_card("normal", "Normal", f"{conf['normal']:.1%} confidence · {conf_word(conf['normal'])}",
                                "Nothing here suggests a tumor — but this tool misses one film in ten, so keep up routine care.")
        return overlay, conf, text

    probB = specialist_probs(rgb, tta, vote)
    idx = int(probB.argmax())
    label = ["benign", "malignant"][idx]
    modelB.zero_grad()
    logitsB = modelB(tfB(rgb).unsqueeze(0).to(device))
    overlay = gradcam_overlay(rgb, featB, gradB, logitsB[0, idx], IB)
    conf = {"benign": float(probB[0]), "malignant": float(probB[1])}
    kind = "non-cancerous" if idx == 0 else "cancerous"
    nxt = ("Save this page and show it to a doctor promptly." if idx == 1
           else "Save this page and show it to a doctor at your next visit.")
    text = verdict_card(label, label.capitalize(),
                        f"{conf[label]:.1%} confidence · {conf_word(conf[label])} · {kind} · tumor probability {p_tumor:.1%}",
                        nxt)
    return overlay, conf, text


def quick(img: Image.Image):
    return predict(img, False, False, 0.5)


def thorough(img: Image.Image):
    return predict(img, True, modelB2 is not None, 0.35)


with gr.Blocks(title="Bone cancer detector — screening aid") as demo:
    gr.HTML("""
    <div id="masthead">
      <p class="eyebrow">Screening aid · proof of concept</p>
      <h1>A second look at bone X-rays.</h1>
      <p class="lede">Two models read every film: one asks <strong>tumor or not</strong>,
      the other asks <strong>harmless or cancerous</strong>. On 564 test films it got
      <strong>505 right</strong> and caught <strong>163 of 179 cancers</strong>.</p>
      <div class="figures">
        <div><b>505 / 564</b><span>test films read correctly</span></div>
        <div><b>163 / 179</b><span>cancers caught</span></div>
        <div><b>96%</b><span>tumor gate accuracy</span></div>
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
                    tbtn = gr.Button("Thorough check — careful + second opinion", variant="secondary")
                tta_box = gr.Checkbox(label="Careful reading — 5 views, slower", value=False)
                ens_box = gr.Checkbox(label="Second opinion — B11 + B12 vote", value=False,
                                      visible=modelB2 is not None)
                thresh = gr.Slider(0.2, 0.7, value=0.5, step=0.05,
                                   label="Sensitivity — lower catches subtler tumors, more false alarms")
        with gr.Column(elem_id="reading"):
            out_img = gr.Image(label="Where the model looked")
            out_label = gr.Label(label="Reading")
            out_text = gr.HTML(label="Verdict")
    gr.HTML("""
    <div id="colophon">
      <p><strong>Not a medical device.</strong> Wrong about one film in ten. A missed tumor
      costs more than a false alarm, so the sensitivity slider leans toward catching.
      Method: EfficientNet-B3 gate, EfficientNet-B6 specialist, Grad-CAM evidence.
      Always see a doctor.</p>
    </div>""")
    btn.click(predict, inputs=[inp, tta_box, ens_box, thresh], outputs=[out_img, out_label, out_text])
    qbtn.click(quick, inputs=[inp], outputs=[out_img, out_label, out_text])
    tbtn.click(thorough, inputs=[inp], outputs=[out_img, out_label, out_text])
    inp.change(predict, inputs=[inp, tta_box, ens_box, thresh], outputs=[out_img, out_label, out_text])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=True, css=CSS)
