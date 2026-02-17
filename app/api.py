import io
import json
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from torchvision import transforms, models

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

    return {
        "prediction": classes[pred.item()],
        "confidence": float(conf.item())
    }
