"""
DR Staging Demo - Flask Backend
Accepts: fundus image, OCT image, or both — independently.
Run: pip install flask flask-cors torch timm einops pillow numpy
Then: python app.py
"""

import io
import os
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "maxvit_tiny_tf_384.in1k"
IMG_SIZE   = 384

FUNDUS_PTH = Path(__file__).parent / "best_fundus_hybrid.pth"
OCT_PTH    = Path(__file__).parent / "best_oct_hybrid.pth"

DR_LABELS  = ["No DR", "NPDR (Non-Proliferative)", "PDR (Proliferative)"]
DME_LABELS = ["No DME", "DME Present"]

DR_DESC = {
    0: "No signs of diabetic retinopathy detected.",
    1: "Non-Proliferative Diabetic Retinopathy — early to moderate stage. Monitoring recommended.",
    2: "Proliferative Diabetic Retinopathy — advanced stage. Urgent specialist referral advised.",
}
DME_DESC = {
    0: "No Diabetic Macular Edema detected.",
    1: "Diabetic Macular Edema detected. Specialist evaluation recommended.",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Model ─────────────────────────────────────────────────────────────────────
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)


class Hybrid_Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            MODEL_NAME, pretrained=False, num_classes=0, global_pool=""
        )
        self.gem = GeM()
        dim = self.backbone.num_features
        self.head_dme = nn.Sequential(nn.Dropout(0.5), nn.Linear(dim, 2))
        self.head_dr  = nn.Sequential(nn.Dropout(0.5), nn.Linear(dim, 3))

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.gem(feat).flatten(1)
        return self.head_dme(feat), self.head_dr(feat)


def load_model(path: Path) -> Hybrid_Expert:
    model = Hybrid_Expert().to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


# ── Pre-processing ────────────────────────────────────────────────────────────
def preprocess(img_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return tensor.to(DEVICE)


@torch.no_grad()
def run_model(model: Hybrid_Expert, img_bytes: bytes) -> dict:
    x = preprocess(img_bytes)
    p_dme1, p_dr1 = model(x)
    p_dme2, p_dr2 = model(torch.flip(x, [3]))

    dr_probs  = ((p_dr1.softmax(1)  + p_dr2.softmax(1))  / 2).squeeze().cpu().tolist()
    dme_probs = ((p_dme1.softmax(1) + p_dme2.softmax(1)) / 2).squeeze().cpu().tolist()

    dr_idx  = int(np.argmax(dr_probs))
    dme_idx = int(np.argmax(dme_probs))

    return {
        "dr": {
            "label":       DR_LABELS[dr_idx],
            "grade":       dr_idx,
            "confidence":  round(dr_probs[dr_idx] * 100, 1),
            "description": DR_DESC[dr_idx],
            "probs":       [round(p * 100, 1) for p in dr_probs],
        },
        "dme": {
            "label":       DME_LABELS[dme_idx],
            "present":     bool(dme_idx),
            "confidence":  round(dme_probs[dme_idx] * 100, 1),
            "description": DME_DESC[dme_idx],
            "probs":       [round(p * 100, 1) for p in dme_probs],
        },
    }


# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static")
CORS(app)

print("Loading models…")
try:
    fundus_model = load_model(FUNDUS_PTH)
    oct_model    = load_model(OCT_PTH)
    print(f"✓ Models loaded on {DEVICE}")
    MODELS_READY = True
except Exception as e:
    print(f"✗ Could not load models: {e}")
    MODELS_READY = False


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/status")
def status():
    return jsonify({"ready": MODELS_READY, "device": DEVICE})


@app.route("/predict", methods=["POST"])
def predict_route():
    if not MODELS_READY:
        return jsonify({"error": "Models not loaded. Check server logs."}), 503

    fundus_file = request.files.get("fundus")
    oct_file    = request.files.get("oct")

    if not fundus_file and not oct_file:
        return jsonify({"error": "Please upload at least one image (fundus or OCT)."}), 400

    # Validate files have content
    if fundus_file and fundus_file.filename == "":
        return jsonify({"error": "Fundus file has no name. Re-upload the file."}), 400
    if oct_file and oct_file.filename == "":
        return jsonify({"error": "OCT file has no name. Re-upload the file."}), 400

    result = {"device": DEVICE}
    try:
        if fundus_file:
            result["fundus"] = run_model(fundus_model, fundus_file.read())
        if oct_file:
            result["oct"] = run_model(oct_model, oct_file.read())
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)