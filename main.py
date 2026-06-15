# ============================================================
# DermAI - FastAPI Backend
# Multi-Model Support: EfficientNet-B0/B4, ResNet50, MobileNetV3
# + Ensemble Mode
# 11 Classes (10 Skin Diseases + Normal Skin)
# ============================================================

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
import io
import os
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dotenv import load_dotenv

# Load secrets from .env file (must be in same folder as main.py)
load_dotenv()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# ============================================================
# 1. CONFIG
# ============================================================
DERMAI_DIR  = r"C:\Users\M Shahid Asghar\Disease Dectection Project\DermAI"
NUM_CLASSES = 11
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Project / Academic Information ──
PROJECT_INFO = {
    "project_title" : "DermAI - AI-Based Skin Disease Detection System",
    "developed_by"  : "Muhammad Shahid Asghar",
    "supervised_by" : "Dr. Akmal Khan",
    "department"    : "Department of Data Science",
    "university"    : "The Islamia University of Bahawalpur (IUB)",
    "province"      : "Punjab",
    "country"       : "Pakistan",
    "team"          : "Individual Project (No Team) — developed solely by Muhammad Shahid Asghar",
    "note"          : (
        "This project was developed as an individual effort by Muhammad Shahid Asghar, "
        "a student of the Department of Data Science, The Islamia University of Bahawalpur (IUB), "
        "Punjab, Pakistan, under the supervision of Dr. Akmal Khan."
    ),
}

# ── Available Models ──
# Jo model train ho chuka ho uska path yahan set karo
MODEL_FILES = {
    "efficientnet_b0"       : os.path.join(DERMAI_DIR, "best_efficientnet_b0.pth"),
    "efficientnet_b4"       : os.path.join(DERMAI_DIR, "best_efficientnet_b4.pth"),
    "resnet50"              : os.path.join(DERMAI_DIR, "best_resnet50.pth"),
    "mobilenetv3_large_100" : os.path.join(DERMAI_DIR, "best_mobilenetv3.pth"),
}

# ── Ensemble Config (weights) ──
ENSEMBLE_WEIGHTS = {
    "efficientnet_b0"       : 0.25,
    "efficientnet_b4"       : 0.40,
    "resnet50"              : 0.20,
    "mobilenetv3_large_100" : 0.15,
}

# ── Image sizes per model ──
MODEL_IMG_SIZES = {
    "efficientnet_b0"       : 224,
    "efficientnet_b4"       : 224,
    "resnet50"              : 224,
    "mobilenetv3_large_100" : 224,
}

CLASS_NAMES = [
    "Eczema",
    "Warts / Molluscum",
    "Melanoma",
    "Atopic Dermatitis",
    "Basal Cell Carcinoma",
    "Melanocytic Nevi",
    "Benign Keratosis",
    "Psoriasis / Lichen",
    "Seborrheic Keratoses",
    "Tinea / Ringworm",
    "Normal Skin",
]

URDU_NAMES = {
    "Eczema"               : "ایگزیما",
    "Warts / Molluscum"    : "مسے / وائرل انفیکشن",
    "Melanoma"             : "میلانوما (جلد کا کینسر)",
    "Atopic Dermatitis"    : "ایٹوپک ڈرمیٹائٹس",
    "Basal Cell Carcinoma" : "بیسل سیل کارسینوما",
    "Melanocytic Nevi"     : "میلانوسٹک نیوی (تل)",
    "Benign Keratosis"     : "بینائن کیراٹوسس",
    "Psoriasis / Lichen"   : "چنبل / لائیکن",
    "Seborrheic Keratoses" : "سیبوریک کیراٹوسس",
    "Tinea / Ringworm"     : "داد / فنگل انفیکشن",
    "Normal Skin"          : "نارمل جلد",
}

SEVERITY_INFO = {
    "Eczema"               : ("Moderate",  "#f59e0b", "Monitor regularly. Use prescribed moisturizers. Consult dermatologist if it worsens."),
    "Warts / Molluscum"    : ("Low",       "#10b981", "Usually harmless. Can be treated or left to resolve naturally. Consult a doctor."),
    "Melanoma"             : ("HIGH RISK", "#ef4444", "URGENT: Please consult a dermatologist or oncologist immediately. Early detection is critical."),
    "Atopic Dermatitis"    : ("Moderate",  "#f59e0b", "Manage with moisturizers and prescribed steroids. Avoid triggers. See a dermatologist."),
    "Basal Cell Carcinoma" : ("HIGH RISK", "#ef4444", "URGENT: Please consult a dermatologist immediately. This is a skin cancer requiring treatment."),
    "Melanocytic Nevi"     : ("Low",       "#10b981", "Usually benign. Monitor for changes in size, shape, or color. Annual skin check recommended."),
    "Benign Keratosis"     : ("Low",       "#10b981", "Non-cancerous growth. Generally harmless but consult a doctor if it changes or irritates."),
    "Psoriasis / Lichen"   : ("Moderate",  "#f59e0b", "Chronic condition. Use prescribed treatments. Consult a dermatologist for management plan."),
    "Seborrheic Keratoses" : ("Low",       "#10b981", "Benign skin growth. No treatment needed unless irritated. Consult doctor if unsure."),
    "Tinea / Ringworm"     : ("Moderate",  "#f59e0b", "Fungal infection. Use antifungal creams. Keep area clean and dry. See doctor if persistent."),
    "Normal Skin"          : ("Normal",    "#10b981", "No skin disease detected. Maintain good skincare routine and consult a dermatologist for regular checkups."),
}

DESCRIPTIONS = {
    "Eczema"               : "Eczema is a chronic inflammatory skin condition causing itchy, red, and dry patches. It commonly appears on hands, neck, and inside the elbows.",
    "Warts / Molluscum"    : "Viral skin infections causing small, rough growths. Warts are caused by HPV, while molluscum is caused by a poxvirus. Both are contagious.",
    "Melanoma"             : "Melanoma is the most dangerous form of skin cancer, developing from pigment-producing cells. Early detection is critical for successful treatment.",
    "Atopic Dermatitis"    : "A chronic form of eczema associated with immune system dysfunction. Causes intense itching, redness, and inflammation.",
    "Basal Cell Carcinoma" : "The most common type of skin cancer. It grows slowly and rarely spreads but requires prompt medical treatment.",
    "Melanocytic Nevi"     : "Commonly known as moles, these are benign skin growths formed by clusters of pigmented cells. Monitor for any changes in appearance.",
    "Benign Keratosis"     : "Non-cancerous skin growths. They appear as waxy, scaly, or rough patches on the skin.",
    "Psoriasis / Lichen"   : "Psoriasis is a chronic autoimmune condition causing rapid skin cell buildup. Lichen planus causes itchy, flat-topped bumps.",
    "Seborrheic Keratoses" : "Common benign skin growths that appear as waxy, brown, or black spots. They are harmless but can be mistaken for melanoma.",
    "Tinea / Ringworm"     : "Fungal infections of the skin, hair, or nails. Treated with antifungal medications.",
    "Normal Skin"          : "No skin disease detected in this image. The skin appears healthy and normal.",
}

# ============================================================
# 2. MODEL ARCHITECTURE
# ============================================================
class SkinModel(nn.Module):
    def __init__(self, model_name, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        self.dropout  = nn.Dropout(0.4)
        # FIX: dummy forward pass to get ACTUAL output features
        # backbone.num_features is wrong for MobileNetV3 (960 vs actual 1280)
        self.backbone.eval()
        with torch.no_grad():
            dummy           = torch.zeros(1, 3, 224, 224)
            actual_features = self.backbone(dummy).shape[1]
        self.backbone.train()
        self.classifier = nn.Linear(actual_features, num_classes)

    def forward(self, x):
        return self.classifier(self.dropout(self.backbone(x)))

# ============================================================
# 3. LOAD ALL AVAILABLE MODELS
# ============================================================
loaded_models  = {}   # { model_name: model }
loaded_info    = {}   # { model_name: {auc, epoch, ...} }

print(f"\nDevice: {DEVICE}")
print("=" * 55)
print("  Loading Available Models...")
print("=" * 55)

for model_name, model_path in MODEL_FILES.items():
    if not os.path.exists(model_path):
        print(f"  SKIP : {model_name} — file not found: {model_path}")
        continue
    try:
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

        # Get num_classes from checkpoint or use default
        n_classes = checkpoint.get('num_classes', NUM_CLASSES)

        model = SkinModel(model_name=model_name, num_classes=n_classes).to(DEVICE)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()

        loaded_models[model_name] = model
        loaded_info[model_name] = {
            "auc"        : round(checkpoint.get('auc', 0), 4),
            "accuracy"   : round(checkpoint.get('accuracy', 0), 2),
            "epoch"      : checkpoint.get('epoch', 0),
            "num_classes": n_classes,
            "file_size"  : round(file_size, 1),
        }
        print(f"  OK   : {model_name:<30} AUC={checkpoint.get('auc',0):.4f}  Ep={checkpoint.get('epoch','?')}  ({file_size:.1f}MB)")

    except Exception as e:
        print(f"  FAIL : {model_name} — {e}")

print(f"\n  Loaded {len(loaded_models)} model(s): {list(loaded_models.keys())}")

# ── Check ensemble config ──
ensemble_path = os.path.join(DERMAI_DIR, "ensemble_config.json")
ensemble_config = None
if os.path.exists(ensemble_path):
    try:
        with open(ensemble_path) as f:
            ensemble_config = json.load(f)
        print(f"  Ensemble config found: {ensemble_path}")
    except:
        pass

print("=" * 55)

# ── Pick default model (best AUC among loaded) ──
if loaded_models:
    DEFAULT_MODEL = max(loaded_info, key=lambda k: loaded_info[k]['auc'])
    print(f"  Default model : {DEFAULT_MODEL} (AUC={loaded_info[DEFAULT_MODEL]['auc']})")
else:
    DEFAULT_MODEL = None
    print("  WARNING: No models loaded! API will return errors.")

# ============================================================
# 4. PREPROCESSING
# ============================================================
def get_transform(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def preprocess_image(image_bytes: bytes, img_size: int = 224) -> torch.Tensor:
    pil_img  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_img   = np.array(pil_img)
    transform = get_transform(img_size)
    return transform(image=np_img)['image'].unsqueeze(0).to(DEVICE)

# ============================================================
# 5. PREDICTION HELPERS
# ============================================================
def predict_single(model_name: str, image_bytes: bytes):
    """Single model prediction."""
    model    = loaded_models[model_name]
    img_size = MODEL_IMG_SIZES.get(model_name, 224)
    tensor   = preprocess_image(image_bytes, img_size)

    with torch.no_grad():
        outputs  = model(tensor)
        probs    = torch.softmax(outputs, dim=1)[0]
        probs_np = probs.cpu().numpy()

    return probs_np


def predict_ensemble(image_bytes: bytes):
    """Weighted ensemble of all loaded models."""
    ensemble_probs = None
    total_weight   = 0.0

    for model_name, model in loaded_models.items():
        weight   = ENSEMBLE_WEIGHTS.get(model_name, 1.0 / len(loaded_models))
        img_size = MODEL_IMG_SIZES.get(model_name, 224)
        tensor   = preprocess_image(image_bytes, img_size)

        with torch.no_grad():
            outputs  = model(tensor)
            probs    = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        if ensemble_probs is None:
            ensemble_probs = weight * probs
        else:
            ensemble_probs += weight * probs
        total_weight += weight

    # Normalize
    ensemble_probs = ensemble_probs / total_weight
    return ensemble_probs


def build_response(probs_np, model_used="single", models_used=None):
    """Build standard JSON response from probability array."""
    # Determine actual number of classes from probs
    n = len(probs_np)
    class_names = CLASS_NAMES[:n]

    top3_indices = probs_np.argsort()[::-1][:3]
    top3 = [
        {
            "rank"       : i + 1,
            "name"       : class_names[idx],
            "confidence" : round(float(probs_np[idx]) * 100, 2),
        }
        for i, idx in enumerate(top3_indices)
    ]

    best_idx   = int(probs_np.argmax())
    best_class = class_names[best_idx]
    confidence = round(float(probs_np[best_idx]) * 100, 2)

    severity, severity_color, advice = SEVERITY_INFO[best_class]

    if confidence < 60:
        advice = "This result is inconclusive (low confidence). Please consult a qualified dermatologist for proper diagnosis."

    all_probs = {
        class_names[i]: round(float(probs_np[i]) * 100, 2)
        for i in range(n)
    }

    return {
        "predicted_class" : best_class,
        "urdu_name"       : URDU_NAMES[best_class],
        "confidence"      : confidence,
        "severity"        : severity,
        "severity_color"  : severity_color,
        "description"     : DESCRIPTIONS[best_class],
        "advice"          : advice,
        "model_used"      : model_used,
        "models_available": list(loaded_models.keys()),
        "models_used"     : models_used or [model_used],
        "top3"            : top3,
        "all_probs"       : all_probs,
    }

# ============================================================
# 6. FASTAPI APP
# ============================================================
app = FastAPI(
    title       = "DermAI — Skin Disease Detection API",
    description = (
        "Multi-Model: EfficientNet-B0/B4, ResNet50, MobileNetV3 + Ensemble. "
        "Developed by Muhammad Shahid Asghar, Department of Data Science, "
        "The Islamia University of Bahawalpur (IUB), Punjab, Pakistan, "
        "under the supervision of Dr. Akmal Khan (Individual Project — No Team)."
    ),
    version     = "2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ============================================================
# 7. ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "message"         : "DermAI API v2.0 is running!",
        "project_info"    : PROJECT_INFO,
        "models_loaded"   : len(loaded_models),
        "models_available": list(loaded_models.keys()),
        "models_info"     : loaded_info,
        "default_model"   : DEFAULT_MODEL,
        "num_classes"     : NUM_CLASSES,
        "device"          : str(DEVICE),
        "docs"            : "/docs",
        "endpoints"       : {
            "POST /predict"          : "Auto-select best model",
            "POST /predict/{model}"  : "Use specific model",
            "POST /predict/ensemble" : "Use all loaded models (weighted)",
            "GET  /models"           : "List all loaded models",
            "GET  /health"           : "Health check",
            "GET  /about"            : "Project & academic information",
        }
    }


@app.get("/health")
def health():
    return {
        "status"          : "ok",
        "device"          : str(DEVICE),
        "models_loaded"   : len(loaded_models),
        "models_available": list(loaded_models.keys()),
        "default_model"   : DEFAULT_MODEL,
    }


@app.get("/models")
def list_models():
    """List all available loaded models with their info."""
    return {
        "models_loaded"   : len(loaded_models),
        "default_model"   : DEFAULT_MODEL,
        "models"          : loaded_info,
        "ensemble_ready"  : len(loaded_models) >= 2,
    }


@app.get("/about")
def about():
    """Project, university, department, and academic supervision information."""
    return {
        "project_title" : PROJECT_INFO["project_title"],
        "developed_by"  : PROJECT_INFO["developed_by"],
        "supervised_by" : PROJECT_INFO["supervised_by"],
        "department"    : PROJECT_INFO["department"],
        "university"    : PROJECT_INFO["university"],
        "province"      : PROJECT_INFO["province"],
        "country"       : PROJECT_INFO["country"],
        "team"          : PROJECT_INFO["team"],
        "description_en": PROJECT_INFO["note"],
        "description_ur": (
            "یہ پراجیکٹ محمد شاہد اصغر نے، شعبہ ڈیٹا سائنس، اسلامیہ یونیورسٹی بہاولپور (IUB)، "
            "صوبہ پنجاب، پاکستان میں، ڈاکٹر اکمل خان کی نگرانی میں اکیلے تیار کیا ہے۔ "
            "اس پراجیکٹ میں کوئی دوسری ٹیم شامل نہیں ہے۔"
        ),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Auto-select: Uses best AUC model if only 1 loaded,
    uses ensemble if 2+ models loaded.
    """
    if not loaded_models:
        raise HTTPException(status_code=503, detail="No models loaded. Please train and save models first.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed.")

    image_bytes = await file.read()
    if len(image_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large. Max 15MB.")

    try:
        if len(loaded_models) >= 2:
            # Ensemble mode
            probs_np  = predict_ensemble(image_bytes)
            model_used = "ensemble"
            models_used = list(loaded_models.keys())
        else:
            # Single best model
            probs_np  = predict_single(DEFAULT_MODEL, image_bytes)
            model_used = DEFAULT_MODEL
            models_used = [DEFAULT_MODEL]

        response = build_response(probs_np, model_used, models_used)
        return JSONResponse(response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/{model_name}")
async def predict_with_model(model_name: str, file: UploadFile = File(...)):
    """
    Use a specific model by name.
    model_name: efficientnet_b0 | efficientnet_b4 | resnet50 | mobilenetv3_large_100 | ensemble
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed.")

    image_bytes = await file.read()
    if len(image_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large. Max 15MB.")

    try:
        if model_name == "ensemble":
            if len(loaded_models) < 2:
                raise HTTPException(status_code=400, detail=f"Ensemble needs 2+ models. Currently loaded: {list(loaded_models.keys())}")
            probs_np   = predict_ensemble(image_bytes)
            model_used = "ensemble"
            models_used = list(loaded_models.keys())

        elif model_name in loaded_models:
            probs_np   = predict_single(model_name, image_bytes)
            model_used = model_name
            models_used = [model_name]

        else:
            available = list(loaded_models.keys()) + ["ensemble"]
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not loaded. Available: {available}"
            )

        response = build_response(probs_np, model_used, models_used)
        return JSONResponse(response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ============================================================
# 8. RUN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)



# ============================================================
# VALIDATE IMAGE ENDPOINT
# ============================================================
@app.post("/validate-image")
async def validate_image(file: UploadFile = File(...)):
    """
    Basic image validation — checks if uploaded file is a valid image.
    Always returns is_skin=True to allow all images through.
    (Advanced skin detection requires separate ML model)
    """
    if not file.content_type.startswith("image/"):
        return JSONResponse({
            "is_skin"   : False,
            "message"   : "Please upload an image file (JPG, PNG, etc.)",
            "message_ur": "براہ کرم تصویر اپلوڈ کریں (JPG, PNG وغیرہ)",
            "confidence": 0.0,
        })

    image_bytes = await file.read()
    if len(image_bytes) < 1000:
        return JSONResponse({
            "is_skin"   : False,
            "message"   : "Image is too small or corrupted. Please upload a clear skin image.",
            "message_ur": "تصویر بہت چھوٹی یا خراب ہے۔ براہ کرم واضح تصویر اپلوڈ کریں۔",
            "confidence": 0.0,
        })

    # Allow all valid images through
    return JSONResponse({
        "is_skin"   : True,
        "message"   : "Valid image detected.",
        "message_ur": "تصویر درست ہے۔",
        "confidence": 1.0,
    })

# ============================================================
# 9. GROQ CHAT ENDPOINT  (Free — llama-3.3-70b)
# ============================================================
from pydantic import BaseModel
from typing import List, Optional
import httpx

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "llama-3.3-70b-versatile"
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not found in environment (.env file).")
    print("         The /chat endpoint will not work until you set it.")

SYSTEM_EN = """You are DermAI Assistant, an AI medical health assistant for the DermAI platform.
DermAI was developed by Muhammad Shahid Asghar, a student of the Department of Data Science, The Islamia University of Bahawalpur (IUB), Punjab, Pakistan, under the supervision of Dr. Akmal Khan. This is an individual project — there is no team involved, it was built entirely by Muhammad Shahid Asghar.
Your specialty is skin diseases — Eczema, Warts/Molluscum, Melanoma, Atopic Dermatitis, Basal Cell Carcinoma, Melanocytic Nevi, Benign Keratosis, Psoriasis/Lichen, Seborrheic Keratoses, Tinea/Ringworm, Normal Skin — and you should give detailed, confident answers on these.
You can also answer general medical and health-related questions on ANY disease, condition, symptom, medicine, or wellness topic (e.g. diabetes, fever, blood pressure, nutrition, mental health, first aid, etc.) — give clear, accurate, patient-friendly information.
Guidelines:
1. Give clear, patient-friendly explanations for any medical/health question asked
2. Mention common symptoms, causes, and general treatment/management options
3. Recommend urgent doctor consultation for any serious, emergency, or high-risk condition (e.g. Melanoma, BCC, chest pain, difficulty breathing, severe bleeding)
4. Keep responses concise (3-6 sentences)
5. If a question is completely unrelated to health/medicine (e.g. coding, sports, politics), politely redirect the user back to health-related topics
6. If asked who made/developed this project or about its background, explain that it was developed by Muhammad Shahid Asghar, Department of Data Science, The Islamia University of Bahawalpur, Punjab, Pakistan, under the supervision of Dr. Akmal Khan, as an individual project with no team
7. Always end with: ⚠️ This is for educational purposes only. Always consult a qualified doctor for diagnosis and treatment."""

SYSTEM_UR = """آپ DermAI اسسٹنٹ ہیں، DermAI پلیٹ فارم کے AI میڈیکل ہیلتھ اسسٹنٹ ہیں۔
DermAI کو محمد شاہد اصغر نے، شعبہ ڈیٹا سائنس، اسلامیہ یونیورسٹی بہاولپور (IUB)، صوبہ پنجاب، پاکستان میں، ڈاکٹر اکمل خان کی نگرانی میں تیار کیا ہے۔ یہ ایک انفرادی پراجیکٹ ہے — اس میں کوئی ٹیم شامل نہیں، یہ مکمل طور پر محمد شاہد اصغر نے تیار کیا ہے۔
آپ کی خاصیت جلد کی بیماریاں ہیں — ایگزیما، مسے، میلانوما، ایٹوپک ڈرمیٹائٹس، بیسل سیل کارسینوما، تل، بینائن کیراٹوسس، چنبل، سیبوریک کیراٹوسس، داد، نارمل جلد — ان پر تفصیلی جواب دیں۔
اس کے علاوہ آپ کسی بھی بیماری، علامت، دوا، یا صحت سے متعلق عمومی سوال کا بھی جواب دے سکتے ہیں (مثلاً ذیابیطس، بخار، بلڈ پریشر، خوراک، ذہنی صحت، فرسٹ ایڈ وغیرہ)۔
ہدایات:
1. ہر طبی/صحت سے متعلق سوال کا واضح اور آسان جواب دیں
2. علامات، وجوہات اور عمومی علاج بیان کریں
3. سنگین یا ایمرجنسی حالات میں فوری ڈاکٹر سے رجوع کرنے کا مشورہ دیں
4. مختصر جواب دیں (3-6 جملے)
5. اگر سوال صحت سے بالکل غیر متعلق ہو (جیسے کوڈنگ، سیاست، کھیل)، تو نرمی سے صحت کے موضوع کی طرف رہنمائی کریں
6. اگر کوئی پوچھے کہ یہ پراجیکٹ کس نے بنایا ہے یا اس کے بارے میں، تو بتائیں کہ یہ محمد شاہد اصغر نے، شعبہ ڈیٹا سائنس، اسلامیہ یونیورسٹی بہاولپور، صوبہ پنجاب، پاکستان میں، ڈاکٹر اکمل خان کی نگرانی میں، بغیر کسی ٹیم کے، اکیلے تیار کیا ہے
7. آخر میں لازمی لکھیں: ⚠️ یہ صرف تعلیمی مقاصد کے لیے ہے۔ ہمیشہ مناسب ڈاکٹر سے مشورہ کریں۔"""


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    language: Optional[str] = "en"


@app.post("/chat")
async def chat(request: ChatRequest):
    """Groq llama-3.3-70b powered DermAI chat assistant."""
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfiguration: GROQ_API_KEY is not set. Add it to your .env file.")

    try:
        system_prompt = SYSTEM_UR if request.language == "ur" else SYSTEM_EN

        messages = [{"role": "system", "content": system_prompt}]
        for msg in request.messages[-10:]:
            messages.append({"role": msg.role, "content": msg.content})

        payload = {
            "model"      : GROQ_MODEL,
            "messages"   : messages,
            "max_tokens" : 600,
            "temperature": 0.7,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                GROQ_URL,
                headers={
                    "Content-Type" : "application/json",
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                },
                json=payload
            )

        if response.status_code != 200:
            print(f"Groq error {response.status_code}: {response.text}")
            raise HTTPException(status_code=502, detail=f"Groq API error {response.status_code}: {response.text}")

        result = response.json()
        reply  = result["choices"][0]["message"]["content"]

        return JSONResponse({"reply": reply, "model": GROQ_MODEL, "language": request.language})

    except HTTPException:
        raise
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Groq API timed out. Please try again.")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Cannot reach Groq API: {str(e)}")
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")
