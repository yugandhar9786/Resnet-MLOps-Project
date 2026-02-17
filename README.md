# 🧠 ResNet50 MLOps — Automated Visual Inspection

## 📌 Overview

This project implements a **production-ready image classification system** using ResNet50 to simulate automated defect detection in manufacturing.  
It demonstrates an end-to-end MLOps workflow including API serving, containerization, and Kubernetes/KServe readiness.

---

## 🏭 Business Problem

Manual visual inspection in factories is:

- Slow  
- Expensive  
- Error-prone  
- Hard to scale  

This leads to defective products reaching customers and increased operational costs.

### 🎯 Objective

Build an AI-powered system that can:

- Detect defective products in real time  
- Provide consistent quality checks  
- Scale with production load  
- Operate reliably 24/7  

---

## 🔄 System Flow

1. Product image is captured by camera  
2. Image sent to FastAPI `/predict` endpoint  
3. Image is preprocessed  
4. ResNet50 performs inference  
5. Prediction returned as JSON  
6. Factory system takes action (accept/reject)

---

## 🧱 Tech Stack

- **Model:** ResNet50 (PyTorch)  
- **API:** FastAPI  
- **Containerization:** Docker  
- **Orchestration:** Kubernetes  
- **Model Serving:** KServe  
- **Monitoring:** Prometheus & Grafana  

---

## 🎯 Key KPIs

- Accuracy > 95%  
- Latency < 200 ms  
- High throughput  
- 24/7 availability  

---

An end-to-end image classification inference system built using
**PyTorch, FastAPI, and ResNet18** to detect bottle defects from the
MVTec dataset.

This project demonstrates a **production-style ML pipeline** including
data preparation, model training, and real-time inference.

------------------------------------------------------------------------

# 🚀 Project Overview

This system performs:

-   ✅ Data preparation from raw MVTec dataset\
-   ✅ Transfer learning using ResNet18\
-   ✅ Model artifact generation\
-   ✅ FastAPI inference service\
-   ✅ Real-time image prediction\
-   ✅ Swagger API testing

------------------------------------------------------------------------

# 🏗️ Architecture

    Raw Data → Processed Data → Model Training → Model Artifacts
                                               ↓
    User Image → FastAPI → PyTorch Model → Prediction → JSON Response

------------------------------------------------------------------------

# 📁 Project Structure

    resnet-mlops-project/
    ├── app/
    │   └── api.py              # FastAPI inference service
    ├── src/
    │   └── train.py            # Model training script
    ├── data/
    │   ├── raw/                # Original MVTec dataset
    │   └── processed/          # Classification-ready dataset
    ├── artifacts/
    │   ├── model.pth           # Trained model weights
    │   └── classes.json        # Class mapping
    ├── prepare_dataset.py      # Data preparation script
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

# ⚙️ Setup Instructions

## 1️⃣ Create Virtual Environment

``` bash
python -m venv .venv
source .venv/bin/activate
```

------------------------------------------------------------------------

## 2️⃣ Install Dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 📊 Step 1 --- Prepare Dataset

## Why this step?

The raw MVTec dataset is **not in classification format**.\
We convert it into PyTorch `ImageFolder` structure.

## Run:

``` bash
python prepare_dataset.py
```

## Output:

    data/processed/
    ├── train/
    └── val/

✅ Dataset becomes training-ready.

------------------------------------------------------------------------

# 🧠 Step 2 --- Train the Model

We use **ResNet18 with transfer learning**.

## Run:

``` bash
python src/train.py
```

## What happens:

-   Loads pretrained ResNet18\
-   Trains on bottle dataset\
-   Saves model artifacts

## Output artifacts:

    artifacts/model.pth
    artifacts/classes.json

------------------------------------------------------------------------

# 🔮 Step 3 --- Run Inference API

Start FastAPI server:

``` bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

------------------------------------------------------------------------

# 🌐 Step 4 --- Test via Swagger UI

Open in browser:

    http://localhost:8000/docs

------------------------------------------------------------------------

## Test Prediction

1.  Open **POST /predict**
2.  Click **Try it out**
3.  Upload bottle image
4.  Click **Execute**

------------------------------------------------------------------------

# ✅ Example Output

``` json
{
  "prediction": "contamination",
  "confidence": 0.9987
}
```

------------------------------------------------------------------------

# 🧾 API Endpoints

## Health Check

    GET /health

Response:

``` json
{"status": "ok"}
```

------------------------------------------------------------------------

## Prediction

    POST /predict

**Input:** image file\
**Output:** predicted class + confidence

------------------------------------------------------------------------

# 🧠 Model Details

-   Architecture: ResNet18\

-   Framework: PyTorch\

-   Input size: 224×224\

-   Classes:

    -   good\
    -   broken_large\
    -   broken_small\
    -   contamination

------------------------------------------------------------------------

# 📦 DVC Integration Guide --- Bottle Defect Classification Project

This document explains how to integrate **DVC (Data Version Control)**
into the ResNet MLOps project for proper dataset and model versioning.

DVC helps track large files (datasets, models) while keeping Git
repositories lightweight and reproducible.

------------------------------------------------------------------------

# 🚀 Why Use DVC?

In MLOps pipelines:

-   ❌ Git is not suitable for large datasets
-   ❌ Models are large binary files
-   ❌ Reproducibility becomes difficult

DVC solves this by:

-   ✅ Versioning datasets
-   ✅ Versioning model artifacts
-   ✅ Enabling reproducible pipelines
-   ✅ Supporting remote storage (S3, GDrive, etc.)

------------------------------------------------------------------------

# 🧰 Step-by-Step DVC Implementation

Follow these steps from your project root.

------------------------------------------------------------------------

## 1️⃣ Install DVC

``` bash
pip install dvc
```

Verify:

``` bash
dvc --version
```

------------------------------------------------------------------------

## 2️⃣ Initialize DVC in the Project

Run inside project root:

``` bash
dvc init
```

This creates:

    .dvc/
    .dvcignore

Now commit to Git:

``` bash
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

------------------------------------------------------------------------

## 3️⃣ Track the Raw Dataset

We track the **raw data folder**, not individual files.

``` bash
dvc add data/raw
```

This creates:

    data/raw.dvc

------------------------------------------------------------------------

## 4️⃣ Update .gitignore (Automatic)

DVC automatically adds:

    data/raw

to `.gitignore` so Git does not store large files.

------------------------------------------------------------------------

## 5️⃣ Commit DVC Metadata

``` bash
git add data/raw.dvc .gitignore
git commit -m "Track raw dataset with DVC"
```

------------------------------------------------------------------------

# 🔄 Data Reproducibility Workflow

## Pull data (on new machine)

``` bash
dvc pull
```

## Push data to remote

``` bash
dvc push
```

------------------------------------------------------------------------

# ☁️ (Optional but Recommended) Add Remote Storage

Example using local remote:

``` bash
mkdir -p ~/dvc-storage
dvc remote add -d localstorage ~/dvc-storage
```

Push data:

``` bash
dvc push
```

------------------------------------------------------------------------

# 🧠 Track Processed Data (Optional)

If you want full pipeline reproducibility:

``` bash
dvc add data/processed
git add data/processed.dvc .gitignore
git commit -m "Track processed dataset with DVC"
```

------------------------------------------------------------------------

# 🧠 Track Model Artifacts (Recommended)

``` bash
dvc add artifacts/model.pth
git add artifacts/model.pth.dvc .gitignore
git commit -m "Track trained model with DVC"
```

------------------------------------------------------------------------

# 🔁 Typical Daily Workflow

## After data changes

``` bash
dvc add data/raw
git add data/raw.dvc
git commit -m "Update dataset version"
dvc push
```

------------------------------------------------------------------------

## After model retraining

``` bash
dvc add artifacts/model.pth
git add artifacts/model.pth.dvc
git commit -m "Update model version"
dvc push
```

------------------------------------------------------------------------

# 📈 What This Achieves

With DVC integrated, your project now supports:

-   ✅ Dataset versioning\
-   ✅ Model versioning\
-   ✅ Reproducible ML pipeline\
-   ✅ Lightweight Git repo\
-   ✅ Production-ready MLOps workflow

------------------------------------------------------------------------

# 📈 What This Project Demonstrates

This project showcases **real MLOps fundamentals**:

-   ✅ Data pipeline design\
-   ✅ Transfer learning\
-   ✅ Model artifact management\
-   ✅ REST inference service\
-   ✅ Async file handling\
-   ✅ Production-style API


## 🚀 Outcome

This project demonstrates the ability to take a deep learning model from development to a **scalable, production-grade MLOps deployment**.

---

**Author:** Yugandhar Kanaparthi
