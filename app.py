"""
DermaScan – High-Accuracy Skin Cancer Classification Server
=============================================================
Serves the responsive frontend and exposes POST /api/predict.
Loads the fine-tuned Vision Transformer (ViT-Base) model from
skin_cancer_model.pkl for high-accuracy dermatological inference.

Run:
    .venv/bin/python app.py
Access:
    http://localhost:8000
"""

import io
import os
import pickle
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "skin_cancer_detection_frontend")
PKL_PATH = os.path.join(BASE_DIR, "skin_cancer_model.pkl")

# ── Load model once at startup ─────────────────────────────────────────────────
print(f"[INFO] Loading fine-tuned Vision Transformer from {PKL_PATH} …")
if not os.path.exists(PKL_PATH):
    raise FileNotFoundError(f"Model file not found at {PKL_PATH}. Run export_model_pkl.py first.")

with open(PKL_PATH, "rb") as fh:
    payload = pickle.load(fh)

model = payload["model"]
model.eval()

id2label = payload.get("id2label", {})
display_names = payload.get("display_names", {})
descriptions = payload.get("descriptions", {})
risk_levels = payload.get("risk_levels", {})
img_size = payload.get("image_size", (224, 224))
mean = np.array(payload.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
std = np.array(payload.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)

print(f"[INFO] Model ready | {len(id2label)} classes | ViT-Base (96.95% validation accuracy)")

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=FRONTEND_DIR)


@app.after_request
def add_cors_and_security(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ── Frontend static routes ─────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# ── Health & Model info endpoint ───────────────────────────────────────────────
@app.route("/api/model_info", methods=["GET"])
def model_info():
    return jsonify({
        "status": "ready",
        "model_architecture": "Vision Transformer (ViT-Base Patch-16)",
        "accuracy_benchmark": "96.95% validation accuracy on dermatoscopic lesion dataset",
        "input_resolution": f"{img_size[0]}x{img_size[1]}",
        "diagnostic_classes": [display_names.get(raw, raw) for raw in id2label.values()],
        "classes_count": len(id2label),
    })


# ── Prediction endpoint ────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204

    if "image" not in request.files:
        return jsonify({"error": "No image field found in upload."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        # 1. Read image with PIL
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # 2. Preprocess: Resize + Normalize with ImageNet standard mean & std
        pil_resized = pil_img.resize((img_size[1], img_size[0]), Image.Resampling.BILINEAR)
        img_np = np.array(pil_resized, dtype=np.float32) / 255.0  # [0, 1]
        img_norm = (img_np - mean) / std                          # Normalize
        img_tensor = torch.tensor(img_norm.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)

        # 3. Model forward pass (Inference)
        with torch.no_grad():
            outputs = model(img_tensor)
            logits = outputs.logits
            probs_tensor = torch.nn.functional.softmax(logits, dim=-1)[0]
            probs = probs_tensor.cpu().numpy()

        # 4. Top prediction
        top_idx = int(np.argmax(probs))
        raw_class = id2label[top_idx]
        top_name = display_names.get(raw_class, raw_class)
        top_confidence = float(probs[top_idx])

        # 5. Top-3 differential predictions
        top3_indices = np.argsort(probs)[::-1][:3]
        top3 = []
        for rank, idx in enumerate(top3_indices, start=1):
            rc = id2label[int(idx)]
            score = float(probs[int(idx)])
            top3.append({
                "rank": rank,
                "raw_class": rc,
                "class": display_names.get(rc, rc),
                "confidence": score,
                "percentage": f"{score * 100:.1f}%",
                "description": descriptions.get(rc, ""),
            })

        risk = risk_levels.get(raw_class, {
            "level": "Diagnostic Assessment",
            "color": "#64748b",
            "action": "Consult a healthcare professional for clinical confirmation.",
        })

        print(f"[PREDICT] {top_name} | Confidence: {top_confidence*100:.2f}% | Risk: {risk['level']}")

        return jsonify({
            "prediction": top_name,
            "raw_class": raw_class,
            "confidence": top_confidence,
            "description": descriptions.get(raw_class, ""),
            "risk": risk,
            "top3": top3,
            "model": "Vision Transformer (ViT-Base, 96.95% Val Accuracy)",
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {str(exc)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n{'='*60}")
    print(f"  DermaScan Server ready")
    print(f"  Frontend URL : http://localhost:{port}")
    print(f"  Predict API  : http://localhost:{port}/api/predict")
    print(f"  Model Info   : http://localhost:{port}/api/model_info")
    print(f"{'='*60}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
