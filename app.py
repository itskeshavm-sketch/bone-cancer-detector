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
        text = f"**NORMAL** ({conf['normal']:.1%} confidence)\n\nStage A gate: no tumor found.\n\n{DISCLAIMER}"
        return overlay, conf, text

    probB = specialist_probs(rgb, tta, vote)
    idx = int(probB.argmax())
    label = ["benign", "malignant"][idx]
    modelB.zero_grad()
    logitsB = modelB(tfB(rgb).unsqueeze(0).to(device))
    overlay = gradcam_overlay(rgb, featB, gradB, logitsB[0, idx], IB)
    conf = {"benign": float(probB[0]), "malignant": float(probB[1])}
    kind = "non-cancerous" if idx == 0 else "cancerous"
    text = (f"**{label.upper()}** ({conf[label]:.1%} confidence, {kind})\n\n"
            f"Stage A tumor probability: {p_tumor:.1%}\n\n{DISCLAIMER}")
    return overlay, conf, text


with gr.Blocks(title="🦴 Bone Cancer Detector") as demo:
    gr.Markdown("# 🦴 Bone Cancer Detector — Two-Stage AI Screening Aid\n" + DISCLAIMER)
    gr.Markdown("**Stage A:** tumor-vs-normal gate (96%) → **Stage B:** benign-vs-malignant specialist | "
                "Test accuracy 89.5% (TTA) | Grad-CAM explainability")
    with gr.Row():
        inp = gr.Image(type="pil", label="Upload X-ray")
        out_img = gr.Image(label="Grad-CAM — where the AI looked")
    with gr.Row():
        tta_box = gr.Checkbox(label="Accurate mode — 5-view vote (slower, matches 89.4%)", value=False)
        ens_box = gr.Checkbox(label="Two-expert vote (B11+B12, slower)", value=False,
                              visible=modelB2 is not None)
    thresh = gr.Slider(0.2, 0.7, value=0.5, step=0.05,
                       label="Tumor sensitivity — lower catches subtler tumors, more false alarms")
    out_label = gr.Label(label="Prediction")
    out_text = gr.Markdown()
    btn = gr.Button("Analyze", variant="primary")
    btn.click(predict, inputs=[inp, tta_box, ens_box, thresh], outputs=[out_img, out_label, out_text])
    inp.change(predict, inputs=[inp, tta_box, ens_box, thresh], outputs=[out_img, out_label, out_text])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=True)
