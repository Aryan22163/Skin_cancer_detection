export default function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    return res.status(204).end();
  }

  return res.status(200).json({
    status: "ready",
    model_architecture: "Vision Transformer (ViT-Base Patch-16)",
    accuracy_benchmark: "96.95% validation accuracy on dermatoscopic lesion dataset",
    input_resolution: "224x224",
    diagnostic_classes: [
      "Melanoma",
      "Basal Cell Carcinoma",
      "Actinic Keratosis",
      "Melanocytic Nevus (Mole)",
      "Benign Keratosis (Seborrheic Keratosis)",
      "Vascular Lesion",
      "Dermatofibroma"
    ],
    classes_count: 7
  });
}
