import io
import json
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from torchvision import transforms, models
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram


# =========================
# Config
# =========================
MODEL_PATH = "artifacts/model.pth"
CLASSES_PATH = "artifacts/classes.json"
IMAGE_SIZE = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(
    title="Bottle Defect Classifier",
    version="1.0"
)
Instrumentator().instrument(app).expose(app)

# =========================
# Custom Model Metrics
# =========================
PREDICTION_COUNTER = Counter(
    "model_predictions_total",
    "Total number of predictions",
    ["predicted_class"]
)

CONFIDENCE_HISTOGRAM = Histogram(
    "model_confidence",
    "Prediction confidence distribution"
)

# =========================
# Load classes
# =========================
with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)

num_classes = len(classes)

# =========================
# Load model
# =========================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("✅ Model loaded successfully")

# =========================
# Transform
# =========================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# =========================
# Health Check (VERY IMPORTANT in MLOps)
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}

# =========================
# Prediction Endpoint
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    predicted_class = classes[pred.item()]
    confidence_value = float(conf.item())

    # 🔥 update metrics
    PREDICTION_COUNTER.labels(predicted_class=predicted_class).inc()
    CONFIDENCE_HISTOGRAM.observe(confidence_value)

    return {
        "prediction": predicted_class,
        "confidence": confidence_value,
        "model_version": "v1.0.1"
    }
