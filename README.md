# 🗑️ Smart Waste Segregation System — MobileNetV2 (Edge AI)

> AI-powered waste classification that tells users **where to throw their waste**, not just what it is.

---

## 📁 Project Structure

```
waste-ai/
│
├── waste_classifier_training.ipynb   ← Kaggle training notebook
│
├── model/
│   └── waste_classifier_final.h5    ← Download from Kaggle outputs
│
├── app.py                           ← Flask backend API
├── requirements.txt
├── index.html                       ← Frontend UI
└── README.md
```

---

## 🧠 Architecture

```
Frontend (index.html)
      ↓  fetch POST /api/predict
Flask Backend (app.py)
      ↓  preprocess → predict
MobileNetV2 (fine-tuned)
      ↓
3-class output + Waste Guidance
```

---

## 🟥 Phase 1 — Model Training (Kaggle)

### 1. Open Kaggle
Go to [kaggle.com](https://www.kaggle.com) → New Notebook → Upload `waste_classifier_training.ipynb`

### 2. Add Dataset
- Click **Add Data** → search `garbage classification`
- Add: **Garbage Classification** by asdasdasasdas

### 3. Enable GPU
Notebook Settings → Accelerator → **GPU T4 x2**

### 4. Run All Cells
The notebook will:
- Build train/val dataset (80/20 split)
- Train MobileNetV2 (2 phases: frozen + fine-tune)
- **Save checkpoint every 2 epochs** → `/kaggle/working/checkpoints/`
- Save **best model** → `/kaggle/working/best_model.h5`
- Save **final model** → `/kaggle/working/waste_classifier_final.h5`
- Plot accuracy curves, confusion matrix, per-class accuracy

### 5. Download the Model
From the output panel, download:
- `waste_classifier_final.h5`
- `class_indices.json`

Place `waste_classifier_final.h5` in the `model/` folder.

---

## 🟦 Phase 2 — Backend Setup (VSCode)

```bash
# Clone your GitHub repo
git clone https://github.com/YOUR_USERNAME/waste-ai.git
cd waste-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Place your model
mkdir model
# Copy waste_classifier_final.h5 into ./model/

# Run Flask server
python app.py
```

Server runs at: `http://localhost:5000`

**Health check:** `GET http://localhost:5000/api/health`

---

## 🟩 Phase 3 — Frontend (VSCode)

Open `index.html` in Live Server (VSCode extension) **or** simply open the file directly in your browser.

> Make sure Flask is running on port 5000 before classifying.

---

## 📡 API Reference

### `GET /api/health`
```json
{
  "status": "ok",
  "model_loaded": true,
  "classes": ["organic", "recyclable", "trash"]
}
```

### `POST /api/predict`
**Form-data:** `file` = image file

**Response:**
```json
{
  "success": true,
  "prediction": {
    "class": "recyclable",
    "confidence": 92.3,
    "emoji": "♻️",
    "bin": "Blue Bin",
    "bin_emoji": "🔵",
    "color": "#3b82f6",
    "tip": "Clean and dry before recycling.",
    "do": ["Clean before recycling", "Flatten boxes", "Remove lids"],
    "dont": ["Don't recycle greasy pizza boxes", "Don't bag recyclables"]
  },
  "top3": [
    {"class": "recyclable", "confidence": 92.3},
    {"class": "trash",      "confidence": 5.1},
    {"class": "organic",    "confidence": 2.6}
  ]
}
```

---

## 🏷️ Classes

| Class       | Bin         | Color  |
|-------------|-------------|--------|
| Organic     | Green Bin 🌿 | #22c55e |
| Recyclable  | Blue Bin ♻️  | #3b82f6 |
| Trash       | Black Bin 🗑️ | #64748b |

---

## ✅ Features

- MobileNetV2 transfer learning (Edge AI ready)
- 2-phase training: frozen base → fine-tuning
- Checkpoint every 2 epochs + best model saved
- Accuracy/loss curves + confusion matrix
- Per-class accuracy bar chart
- REST API with smart waste guidance
- Modern dark UI with confidence bars
- Drag & drop image upload
- Top-3 predictions display
- Do/Don't guidance per class

---

## 🚀 GitHub

```bash
git add .
git commit -m "feat: Smart Waste Segregation System — MobileNetV2"
git push origin main
```

**Repo title:** `Smart Waste Segregation System using MobileNet (Edge AI)`