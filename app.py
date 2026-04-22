"""
Smart Waste Segregation System — Flask Backend
"""

import os
import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import io
import base64

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARN] TensorFlow not installed. Running in mock mode.")

# ── Config ─────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, 'model', 'waste_classifier_final.h5')
CLASS_IDX_PATH  = os.path.join(BASE_DIR, 'class_indices.json')
IMG_SIZE        = (224, 224)

# ── Load class index → label mapping ──────────────────────────────────────
if os.path.exists(CLASS_IDX_PATH):
    with open(CLASS_IDX_PATH) as f:
        raw = json.load(f)
    # Invert: {0: "organic", 1: "recyclable", 2: "trash"}
    CLASS_NAMES = {v: k for k, v in raw.items()}
    print(f"[INFO] Class mapping: {CLASS_NAMES}")
else:
    # Fallback — alphabetical default
    CLASS_NAMES = {0: "organic", 1: "recyclable", 2: "trash"}
    print("[WARN] class_indices.json not found. Using default order.")

WASTE_GUIDE = {
    "organic": {
        "emoji"    : "🌿",
        "bin"      : "Green Bin",
        "bin_emoji": "🟢",
        "color"    : "#22c55e",
        "tip"      : "Can be composted at home or in a community garden.",
        "examples" : ["Food scraps", "Vegetable peels", "Fruit rinds", "Coffee grounds", "Yard waste"],
        "do"       : ["Compost at home", "Use municipal organic bin", "Feed to animals if applicable"],
        "dont"     : ["Don't mix with plastic", "Don't put in recycling", "Don't include meat in home compost"]
    },
    "recyclable": {
        "emoji"    : "♻️",
        "bin"      : "Blue Bin",
        "bin_emoji": "🔵",
        "color"    : "#3b82f6",
        "tip"      : "Clean and dry before recycling to avoid contamination.",
        "examples" : ["Plastic bottles", "Cardboard boxes", "Glass jars", "Metal cans", "Paper"],
        "do"       : ["Clean before recycling", "Flatten boxes", "Remove lids from bottles"],
        "dont"     : ["Don't recycle greasy pizza boxes", "Don't bag recyclables in plastic", "Don't include broken glass"]
    },
    "trash": {
        "emoji"    : "🗑️",
        "bin"      : "Black Bin",
        "bin_emoji": "⚫",
        "color"    : "#64748b",
        "tip"      : "Check if your local centre accepts special items like batteries or electronics.",
        "examples" : ["Broken ceramics", "Diapers", "Styrofoam", "Chip bags", "Broken toys"],
        "do"       : ["Dispose in general waste", "Check for special disposal for hazardous items", "Reduce usage"],
        "dont"     : ["Don't put batteries in bin", "Don't pour liquids in bin", "Don't put e-waste in bin"]
    }
}

app   = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
model = None

def load_waste_model():
    global model
    if not TF_AVAILABLE:
        print("[INFO] TF not available — mock mode active.")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] Model not found at {MODEL_PATH}")
        return
    try:
        model = load_model(MODEL_PATH)
        print(f"[INFO] ✅ Model loaded — output shape: {model.output_shape}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")

def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def mock_predict():
    import random
    classes = list(CLASS_NAMES.values())
    random.shuffle(classes)
    a = random.uniform(0.65, 0.95)
    b = random.uniform(0.03, 1 - a - 0.01)
    c = max(0.0, 1 - a - b)
    return [
        {"class": classes[0], "confidence": round(a * 100, 1)},
        {"class": classes[1], "confidence": round(b * 100, 1)},
        {"class": classes[2], "confidence": round(c * 100, 1)},
    ]

@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status"      : "ok",
        "model_loaded": model is not None,
        "tf_available": TF_AVAILABLE,
        "classes"     : list(CLASS_NAMES.values()),
        "class_map"   : CLASS_NAMES
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    img_bytes = None

    if 'file' in request.files:
        img_bytes = request.files['file'].read()
    elif request.is_json:
        data = request.get_json()
        if 'image' in data:
            try:
                header, encoded = data['image'].split(',', 1) if ',' in data['image'] else ('', data['image'])
                img_bytes = base64.b64decode(encoded)
            except Exception as e:
                return jsonify({"error": f"Base64 decode failed: {str(e)}"}), 400

    if img_bytes is None:
        return jsonify({"error": "No image provided."}), 400

    try:
        if model is not None and TF_AVAILABLE:
            arr      = preprocess_image(img_bytes)
            preds    = model.predict(arr, verbose=0)[0]          # shape: (3,)
            top3_idx = np.argsort(preds)[::-1][:3]
            top3 = [
                {
                    "class"     : CLASS_NAMES[int(i)],
                    "confidence": round(float(preds[i]) * 100, 1)
                }
                for i in top3_idx
            ]
        else:
            top3 = mock_predict()

        primary = top3[0]
        guide   = WASTE_GUIDE[primary["class"]]

        return jsonify({
            "success"   : True,
            "prediction": {
                "class"     : primary["class"],
                "confidence": primary["confidence"],
                "emoji"     : guide["emoji"],
                "bin"       : guide["bin"],
                "bin_emoji" : guide["bin_emoji"],
                "color"     : guide["color"],
                "tip"       : guide["tip"],
                "examples"  : guide["examples"],
                "do"        : guide["do"],
                "dont"      : guide["dont"],
            },
            "top3"     : top3,
            "mock_mode": model is None
        }), 200

    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/api/guide', methods=['GET'])
def get_guide():
    return jsonify({"guide": WASTE_GUIDE})

if __name__ == '__main__':
    load_waste_model()
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 Running on http://localhost:{port}")
    app.run(debug=True, host='0.0.0.0', port=port)