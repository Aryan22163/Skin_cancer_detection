"""
DermaScan – High-Accuracy Skin Cancer Model Export
===================================================
Exports a high-accuracy, fine-tuned Vision Transformer (ViT-Base) model
specifically trained on dermatoscopic skin cancer images (96.95% validation accuracy).

Classes:
  - Melanoma (Malignant skin cancer)
  - Basal Cell Carcinoma (Common skin cancer)
  - Actinic Keratoses (Pre-cancerous sun damage lesion)
  - Melanocytic Nevi (Common mole)
  - Benign Keratosis-like Lesions (Seborrheic keratosis, solar lentigo)
  - Vascular Lesions (Angiomas, pyogenic granulomas)
  - Dermatofibroma (Benign cutaneous fibrous histiocytoma)

Outputs:
  - skin_cancer_model.pkl  (Loaded by Flask app for fast, accurate inference)
"""

import os
import pickle
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_PATH = os.path.join(BASE_DIR, "skin_cancer_model.pkl")
MODEL_ID = "Anwarkh1/Skin_Cancer-Image_Classification"

# Human-readable diagnostic class names
DISPLAY_NAMES = {
    "melanoma": "Melanoma",
    "basal_cell_carcinoma": "Basal Cell Carcinoma",
    "actinic_keratoses": "Actinic Keratosis",
    "melanocytic_Nevi": "Melanocytic Nevus (Mole)",
    "benign_keratosis-like_lesions": "Benign Keratosis (Seborrheic Keratosis)",
    "vascular_lesions": "Vascular Lesion",
    "dermatofibroma": "Dermatofibroma",
}

# Clinical descriptions for patient/user education
DESCRIPTIONS = {
    "melanoma": (
        "Malignant skin cancer originating from pigment-producing melanocytes. "
        "Early detection and prompt dermatological evaluation are critical."
    ),
    "basal_cell_carcinoma": (
        "Most common type of skin cancer. Locally invasive but slow-growing, "
        "with extremely high cure rates when excised early."
    ),
    "actinic_keratoses": (
        "Pre-cancerous scaly or crusty patch caused by chronic UV sun exposure. "
        "Should be evaluated to prevent progression to invasive carcinoma."
    ),
    "melanocytic_Nevi": (
        "Benign cluster of melanocyte cells commonly known as a mole. "
        "Harmless, but continue regular skin self-checks using ABCDE rules."
    ),
    "benign_keratosis-like_lesions": (
        "Non-cancerous skin growth such as seborrheic keratosis or solar lentigo. "
        "Common with age, completely benign and harmless."
    ),
    "vascular_lesions": (
        "Benign blood vessel anomaly such as a cherry angioma or pyogenic granuloma. "
        "Non-cancerous vascular formation."
    ),
    "dermatofibroma": (
        "Harmless benign fibrous skin nodule typically occurring on the extremities. "
        "Non-malignant; medical intervention is rarely needed."
    ),
}

# Medical risk severity categorization
RISK_LEVELS = {
    "melanoma": {
        "level": "High Risk (Malignant)",
        "color": "#ef4444",
        "action": "Urgent dermatologist consultation recommended.",
    },
    "basal_cell_carcinoma": {
        "level": "Moderate Risk (Malignant / Local)",
        "color": "#f97316",
        "action": "Schedule evaluation with a dermatologist for diagnosis and removal.",
    },
    "actinic_keratoses": {
        "level": "Pre-Cancerous",
        "color": "#eab308",
        "action": "Consult dermatologist for preventive evaluation and treatment.",
    },
    "melanocytic_Nevi": {
        "level": "Benign (Harmless)",
        "color": "#22c55e",
        "action": "Routine observation; consult if mole changes size, shape, or color.",
    },
    "benign_keratosis-like_lesions": {
        "level": "Benign (Harmless)",
        "color": "#22c55e",
        "action": "Harmless benign lesion; no medical intervention required.",
    },
    "vascular_lesions": {
        "level": "Benign (Harmless)",
        "color": "#22c55e",
        "action": "Harmless vascular lesion; routine observation.",
    },
    "dermatofibroma": {
        "level": "Benign (Harmless)",
        "color": "#22c55e",
        "action": "Benign nodule; no intervention necessary unless uncomfortable.",
    },
}


def export_model():
    print("=" * 65)
    print("  DermaScan – Exporting High-Accuracy Skin Cancer Model")
    print("=" * 65)
    print(f"\n[1/4] Downloading fine-tuned model: {MODEL_ID} …")

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()

    id2label = model.config.id2label
    num_classes = len(id2label)
    print(f"[OK] Model loaded: {num_classes} classes, ViT-Base architecture.")

    # Package into clean pickle payload
    payload = {
        "model": model,
        "model_id": MODEL_ID,
        "model_type": "VisionTransformer",
        "accuracy_benchmark": "96.95% validation accuracy on dermatoscopic skin lesions",
        "image_size": (224, 224),
        "id2label": id2label,
        "display_names": DISPLAY_NAMES,
        "descriptions": DESCRIPTIONS,
        "risk_levels": RISK_LEVELS,
        "mean": processor.image_mean if hasattr(processor, "image_mean") else [0.485, 0.456, 0.406],
        "std": processor.image_std if hasattr(processor, "image_std") else [0.229, 0.224, 0.225],
    }

    print(f"\n[2/4] Saving to pickle: {PKL_PATH} …")
    with open(PKL_PATH, "wb") as f:
        pickle.dump(payload, f)

    size_mb = os.path.getsize(PKL_PATH) / 1024 / 1024
    print(f"[OK] Saved {PKL_PATH} ({size_mb:.1f} MB)")

    print("\n[3/4] Verifying pickle loading & inference test …")
    with open(PKL_PATH, "rb") as f:
        loaded = pickle.load(f)

    test_model = loaded["model"]
    test_model.eval()

    # Test with sample melanoma image if present
    test_image_path = os.path.join(BASE_DIR, "sample_images", "melanoma_sample.jpg")
    if os.path.exists(test_image_path):
        img = Image.open(test_image_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            logits = test_model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            topk = torch.topk(probs, 3)

        print(f"\n[4/4] Verification on real melanoma image ({test_image_path}):")
        for rank, (score, idx) in enumerate(zip(topk.values, topk.indices), 1):
            raw_label = loaded["id2label"][idx.item()]
            name = loaded["display_names"].get(raw_label, raw_label)
            print(f"   #{rank}: {name} -> {score.item()*100:.2f}% confidence")
    else:
        print("[INFO] No test sample image found; model structure verified.")

    print("\n[SUCCESS] Export complete! skin_cancer_model.pkl is ready for use.")


if __name__ == "__main__":
    export_model()
