import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# =========================
# Config
# =========================
DATA_DIR = "data/processed"
MODEL_DIR = "artifacts"
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
IMAGE_SIZE = 224

os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {DEVICE}")

# =========================
# Transforms
# =========================
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# =========================
# Datasets
# =========================
train_dataset = datasets.ImageFolder(
    root=f"{DATA_DIR}/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=f"{DATA_DIR}/val",
    transform=val_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

num_classes = len(train_dataset.classes)
print(f"✅ Classes: {train_dataset.classes}")

# Save class mapping (VERY IMPORTANT for inference)
with open(f"{MODEL_DIR}/classes.json", "w") as f:
    json.dump(train_dataset.classes, f)

# =========================
# Model
# =========================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =========================
# Training Loop
# =========================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"📊 Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

print("✅ Training complete")

# =========================
# Save model
# =========================
MODEL_PATH = f"{MODEL_DIR}/model.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
