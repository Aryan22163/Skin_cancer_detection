// DermaScan front end.
// Connect this to your own backend that loads your trained model.
// Expected API: POST /api/predict with multipart form field "image"
// Expected JSON response: { "prediction": "class name", "confidence": 0.93 }
// Confidence can be a number from 0-1 or 0-100.

const API_ENDPOINT = "/api/predict";

const imageInput = document.getElementById("imageInput");
const browseButton = document.getElementById("browseButton");
const sampleMelanomaBtn = document.getElementById("sampleMelanomaBtn");
const dropZone = document.getElementById("dropZone");
const uploadPlaceholder = document.getElementById("uploadPlaceholder");
const previewWrap = document.getElementById("previewWrap");
const imagePreview = document.getElementById("imagePreview");
const fileName = document.getElementById("fileName");
const removeImage = document.getElementById("removeImage");
const analyzeButton = document.getElementById("analyzeButton");
const statusMessage = document.getElementById("statusMessage");
const emptyResult = document.getElementById("emptyResult");
const resultContent = document.getElementById("resultContent");
const predictionLabel = document.getElementById("predictionLabel");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");
const resultExplanation = document.getElementById("resultExplanation");
const resetButton = document.getElementById("resetButton");

let selectedFile = null;
let previewUrl = null;

if (sampleMelanomaBtn) {
  sampleMelanomaBtn.addEventListener("click", async () => {
    try {
      statusMessage.textContent = "Loading sample melanoma image…";
      const resp = await fetch("/sample_melanoma.jpg");
      if (!resp.ok) throw new Error("Could not load sample image.");
      const blob = await resp.blob();
      const file = new File([blob], "melanoma_sample.jpg", { type: "image/jpeg" });
      setSelectedFile(file);
      statusMessage.textContent = "Sample image loaded! Click 'Analyze image' to run high-accuracy inference.";
    } catch (err) {
      statusMessage.textContent = "Failed to load sample image: " + err.message;
    }
  });
}

browseButton.addEventListener("click", (event) => {
  event.stopPropagation();
  imageInput.click();
});

dropZone.addEventListener("click", (event) => {
  if (event.target.closest("button")) return;
  imageInput.click();
});

dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    imageInput.click();
  }
});

imageInput.addEventListener("change", (event) => {
  const file = event.target.files && event.target.files[0];
  if (file) setSelectedFile(file);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
});

dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files && event.dataTransfer.files[0];
  if (file) setSelectedFile(file);
});

removeImage.addEventListener("click", (event) => {
  event.stopPropagation();
  clearSelectedFile();
});

resetButton.addEventListener("click", () => {
  clearSelectedFile();
  showEmptyResult();
  statusMessage.textContent = "Choose an image to begin.";
});

analyzeButton.addEventListener("click", analyzeImage);

function setSelectedFile(file) {
  const allowedTypes = ["image/jpeg", "image/png", "image/webp"];
  if (!allowedTypes.includes(file.type)) {
    statusMessage.textContent = "Please choose a JPG, PNG, or WEBP image.";
    return;
  }
  const maxBytes = 10 * 1024 * 1024;
  if (file.size > maxBytes) {
    statusMessage.textContent = "The image is too large. Please choose a file under 10 MB.";
    return;
  }

  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  imagePreview.src = previewUrl;
  fileName.textContent = file.name;
  uploadPlaceholder.classList.add("hidden");
  previewWrap.classList.remove("hidden");
  analyzeButton.disabled = false;
  statusMessage.textContent = "Image ready. Run your model when you're ready.";
  showEmptyResult();
}

function clearSelectedFile() {
  selectedFile = null;
  imageInput.value = "";
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  imagePreview.removeAttribute("src");
  uploadPlaceholder.classList.remove("hidden");
  previewWrap.classList.add("hidden");
  analyzeButton.disabled = true;
}

async function analyzeImage() {
  if (!selectedFile) return;

  analyzeButton.disabled = true;
  analyzeButton.querySelector("span:first-child").textContent = "Analyzing…";
  statusMessage.textContent = "Sending image to the model API…";

  try {
    const formData = new FormData();
    formData.append("image", selectedFile);

    const response = await fetch(API_ENDPOINT, {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      throw new Error(`API request failed (${response.status}). Check that your backend is running.`);
    }

    const data = await response.json();
    if (!data || typeof data.prediction !== "string") {
      throw new Error('Unexpected API response. Expected JSON with a "prediction" field.');
    }

    let confidence = Number(data.confidence);
    if (!Number.isFinite(confidence)) confidence = null;
    if (confidence !== null && confidence <= 1) confidence *= 100;
    if (confidence !== null) confidence = Math.max(0, Math.min(100, confidence));

    predictionLabel.textContent = data.prediction;
    confidenceValue.textContent = confidence === null ? "Not provided" : `${confidence.toFixed(1)}%`;
    confidenceBar.style.width = confidence === null ? "0%" : `${confidence}%`;

    // Render clinical risk severity banner
    let riskEl = document.getElementById("riskContainer");
    if (!riskEl) {
      riskEl = document.createElement("div");
      riskEl.id = "riskContainer";
      const track = document.querySelector(".confidence-track");
      if (track) track.after(riskEl);
      else resultContent.appendChild(riskEl);
    }
    if (data.risk) {
      riskEl.innerHTML = `
        <div class="risk-banner">
          <span class="risk-pill" style="background-color: ${data.risk.color || '#64748b'};">${data.risk.level}</span>
          <span class="risk-action">${data.risk.action}</span>
        </div>
      `;
    } else {
      riskEl.innerHTML = "";
    }

    // Show class description if available
    const desc = data.description || "";
    resultExplanation.textContent = desc
      ? desc + " — Model prediction based on dermatoscopic visual patterns. Not a definitive clinical diagnosis."
      : "This is the output of your fine-tuned model, not a confirmed clinical diagnosis.";

    // Render top-3 breakdown if provided
    let top3Html = "";
    if (Array.isArray(data.top3) && data.top3.length > 1) {
      top3Html = `<div class="top3-block"><p class="top3-title">TOP 3 PREDICTIONS</p><ul class="top3-list">`;
      data.top3.forEach((item, i) => {
        const pct = (item.confidence * 100).toFixed(1);
        top3Html += `<li class="top3-item">
          <span class="top3-rank">#${i+1}</span>
          <span class="top3-name">${item.class}</span>
          <div class="top3-bar-wrap"><div class="top3-bar" style="width:${pct}%"></div></div>
          <span class="top3-pct">${pct}%</span>
        </li>`;
      });
      top3Html += `</ul></div>`;
    }
    let top3El = document.getElementById("top3Container");
    if (!top3El) {
      top3El = document.createElement("div");
      top3El.id = "top3Container";
      resultContent.appendChild(top3El);
    }
    top3El.innerHTML = top3Html;

    emptyResult.classList.add("hidden");
    resultContent.classList.remove("hidden");
    statusMessage.textContent = "Model response received.";
    const badge = document.querySelector(".result-badge");
    badge.innerHTML = '<span class="badge-dot active-dot"></span> Result received';
  } catch (error) {
    statusMessage.textContent = error.message || "Could not connect to the model API.";
    // Show an actionable hint without inventing a prediction.
    emptyResult.classList.remove("hidden");
    resultContent.classList.add("hidden");
  } finally {
    analyzeButton.disabled = !selectedFile;
    analyzeButton.querySelector("span:first-child").textContent = "Analyze image";
  }
}

function showEmptyResult() {
  emptyResult.classList.remove("hidden");
  resultContent.classList.add("hidden");
  confidenceBar.style.width = "0%";
  const badge = document.querySelector(".result-badge");
  badge.innerHTML = '<span class="badge-dot"></span> Awaiting image';
  const riskEl = document.getElementById("riskContainer");
  if (riskEl) riskEl.innerHTML = "";
  const top3El = document.getElementById("top3Container");
  if (top3El) top3El.innerHTML = "";
}
