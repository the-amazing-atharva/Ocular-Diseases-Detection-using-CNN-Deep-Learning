import os
import io
import datetime
import json

import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage

from huggingface_hub import hf_hub_download


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Ocular Disease Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# LIGHT / CLEAN CUSTOM CSS  (no dark theme)
# =============================================================================

st.markdown("""
<style>
/* ── overall body ─────────────────────────────────────────── */
.stApp { background-color: #f8fafc; }

/* ── hero banner ──────────────────────────────────────────── */
.hero-wrap {
    background: linear-gradient(120deg, #1e3a5f 0%, #2563eb 60%, #1d4ed8 100%);
    border-radius: 14px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: #fff;
}
.hero-title  { font-size: 2.2rem; font-weight: 800; margin: 0 0 .3rem 0; }
.hero-sub    { font-size: 1rem;   opacity: .85; margin: 0; }
.hero-badge  {
    display: inline-block;
    background: rgba(255,255,255,.18);
    border: 1px solid rgba(255,255,255,.35);
    color: #fff;
    font-size: .7rem;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: .7rem;
}

/* ── metric card ──────────────────────────────────────────── */
.mcard {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.mcard-value { font-size: 1.8rem; font-weight: 700; color: #1e40af; line-height: 1; }
.mcard-label { font-size: .75rem; color: #64748b; text-transform: uppercase;
                letter-spacing: .8px; margin-top: .25rem; }

/* ── result card ──────────────────────────────────────────── */
.rcard {
    background: #fff;
    border: 1.5px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: .6rem;
}
.rcard-disease { border-color: #fca5a5; background: #fff5f5; }
.rcard-normal  { border-color: #86efac; background: #f0fdf4; }
.rcard-title   { font-size: 1.35rem; font-weight: 700; color: #0f172a; }
.rcard-conf    { font-size: .9rem; color: #475569; margin: .2rem 0 .5rem; }

/* ── confidence bar ───────────────────────────────────────── */
.cbar-bg   { background: #e2e8f0; border-radius: 6px; height: 7px; overflow: hidden; }
.cbar-fill { height: 100%; border-radius: 6px; }
.cbar-blue { background: linear-gradient(90deg,#3b82f6,#60a5fa); }
.cbar-red  { background: linear-gradient(90deg,#ef4444,#f87171); }
.cbar-grn  { background: linear-gradient(90deg,#22c55e,#4ade80); }

/* ── risk badges ──────────────────────────────────────────── */
.badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: .78rem; font-weight: 700; letter-spacing: .4px;
}
.badge-low  { background:#dcfce7; color:#15803d; border:1px solid #86efac; }
.badge-med  { background:#fef9c3; color:#a16207; border:1px solid #fde047; }
.badge-high { background:#fee2e2; color:#dc2626; border:1px solid #fca5a5; }

/* ── follow-up card ──────────────────────────────────────── */
.fu-card {
    background: #fff;
    border: 1.5px solid #bfdbfe;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-top: .8rem;
}
.fu-title { font-size: 1rem; font-weight: 700; color: #1e40af; margin-bottom: .5rem; }
.fu-row   { display:flex; gap:.6rem; align-items:flex-start; margin:.35rem 0; }
.fu-bullet{ width:8px; height:8px; border-radius:50%; margin-top:5px; flex-shrink:0; }
.fu-text  { font-size:.86rem; color:#334155; line-height:1.5; }

/* ── section title ────────────────────────────────────────── */
.sec-title {
    font-size: 1rem; font-weight: 700; color: #1e40af;
    border-left: 4px solid #3b82f6; padding-left: 10px;
    margin: 1rem 0 .6rem;
}

/* ── comparison table ─────────────────────────────────────── */
.ctable { width:100%; border-collapse:collapse; font-size:.88rem; }
.ctable th {
    background:#eff6ff; color:#1e40af; padding:9px 14px;
    text-align:left; font-weight:700;
    border-bottom:2px solid #bfdbfe;
}
.ctable td {
    padding:8px 14px; border-bottom:1px solid #f1f5f9; color:#334155;
}
.ctable tr:hover td { background:#f8fafc; }

/* ── timeline items ───────────────────────────────────────── */
.titem {
    display:flex; gap:.9rem; padding:.5rem 0;
    border-left:2px solid #bfdbfe;
    margin-left:.5rem; padding-left:1rem; position:relative;
}
.titem::before {
    content:''; position:absolute; left:-6px; top:50%;
    transform:translateY(-50%);
    width:9px; height:9px; border-radius:50%;
    background:#3b82f6; border:2px solid #eff6ff;
}
.tlabel { font-size:.72rem; color:#3b82f6; font-weight:700;
          text-transform:uppercase; letter-spacing:.7px; min-width:80px; }
.ttext  { color:#374151; font-size:.88rem; }

/* ── tab styling ──────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap:3px; background:#f1f5f9; padding:4px;
    border-radius:10px;
}
.stTabs [data-baseweb="tab"] {
    border-radius:7px; color:#475569;
    font-size:.82rem; font-weight:500; padding:5px 13px;
}
.stTabs [aria-selected="true"] {
    background:#fff !important; color:#1e40af !important;
    font-weight:700 !important; box-shadow:0 1px 4px rgba(0,0,0,.08) !important;
}

/* ── sidebar ──────────────────────────────────────────────── */
[data-testid="stSidebar"] { background:#f0f4ff; border-right:1px solid #e0e7ff; }

/* ── uploader ─────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background:#fff; border:1.5px dashed #93c5fd; border-radius:10px;
}

/* ── divider ──────────────────────────────────────────────── */
.fdiv {
    border:none; height:1px;
    background:linear-gradient(90deg,transparent,#bfdbfe,transparent);
    margin:1.2rem 0;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# PATHS & CONSTANTS
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_REPO_ID = "AtharvaSalitri/eye-disease-model"
MODEL_PATH = hf_hub_download(repo_id=MODEL_REPO_ID, filename="best_model.h5")

HERO_IMAGE_PATH = os.path.join(
    BASE_DIR, "diabetic-eye-issues-5-ways-diabetes-impacts-vision.jpg")
VISUALIZATION_DIR = os.path.join(BASE_DIR, "Visualizations")
SAMPLE_DIR = os.path.join(BASE_DIR, "Samples")

Class_Names_Dict = {"glaucoma": 0, "normal": 1,
                    "cataract": 2, "diabetic_retinopathy": 3}
class_names = list(Class_Names_Dict.keys())

DISPLAY_NAMES = {
    "glaucoma": "Glaucoma",
    "normal": "Normal",
    "cataract": "Cataract",
    "diabetic_retinopathy": "Diabetic Retinopathy",
}

CLASS_ICONS = {
    "glaucoma": "🔵",
    "normal": "✅",
    "cataract": "⬜",
    "diabetic_retinopathy": "🔴",
}

CLASS_COLORS_MPL = {
    "glaucoma": "#3b82f6",
    "normal": "#22c55e",
    "cataract": "#a855f7",
    "diabetic_retinopathy": "#ef4444",
}

CLASS_RISK = {
    "glaucoma":             ("high",   "Prompt ophthalmology referral recommended"),
    "normal":               ("low",    "Continue routine annual eye exams"),
    "cataract":             ("medium", "Schedule elective ophthalmology consultation"),
    "diabetic_retinopathy": ("high",   "Urgent ophthalmology referral recommended"),
}


# =============================================================================
# DISEASE INFORMATION
# =============================================================================

DISEASE_INFO = {
    "glaucoma": {
        "title": "🔵 Glaucoma",
        "description": (
            "Glaucoma is a group of eye conditions that progressively damage the optic nerve, "
            "usually associated with elevated intraocular pressure. It is the second leading "
            "cause of blindness worldwide. Vision loss is irreversible, making early detection critical."
        ),
        "symptoms": [
            "Gradual loss of peripheral (side) vision",
            "Tunnel vision in advanced stages",
            "Blurred vision and halos around lights",
            "Eye pain or redness (acute angle-closure form)",
            "Nausea associated with eye pain",
        ],
        "risk_factors": [
            "Age > 60 years",
            "Family history of glaucoma",
            "Elevated intraocular pressure",
            "African, Hispanic, or Asian ancestry",
            "Previous eye injury or surgery",
            "Long-term corticosteroid use",
        ],
        "management": (
            "Treatment includes prescription eye drops (prostaglandins, beta-blockers), "
            "laser trabeculoplasty, or surgical intervention to reduce intraocular pressure."
        ),
        "prevalence": "~80 million worldwide",
        "screening_freq": "Every 1–2 years after age 40",
        "icd_code": "H40",
    },
    "normal": {
        "title": "✅ Normal",
        "description": (
            "No disease-specific retinal patterns were detected. A 'Normal' result does not "
            "constitute a clean bill of ocular health — routine exams remain essential."
        ),
        "symptoms": [
            "No disease-specific pattern detected",
            "Normal classification does not guarantee perfect eye health",
            "Some early-stage conditions may not yet be visible",
        ],
        "risk_factors": [
            "All individuals carry some baseline risk",
            "Risk increases substantially after age 40",
            "Systemic conditions (diabetes, hypertension) elevate ocular risk",
        ],
        "management": (
            "Continue routine comprehensive eye examinations annually or as recommended "
            "by your eye-care professional."
        ),
        "prevalence": "Majority of population",
        "screening_freq": "Every 1–2 years",
        "icd_code": "Z01.00",
    },
    "cataract": {
        "title": "⬜ Cataract",
        "description": (
            "A cataract is a clouding of the crystalline lens causing progressive blurring. "
            "It is the leading cause of blindness worldwide but is highly treatable with surgery."
        ),
        "symptoms": [
            "Cloudy or blurry vision",
            "Increased sensitivity to glare",
            "Difficulty seeing at night",
            "Faded or yellowed colours",
            "Frequent prescription changes",
            "Double vision in one eye",
        ],
        "risk_factors": [
            "Advancing age (most common after 60)",
            "Diabetes mellitus",
            "Prolonged UV light exposure",
            "Smoking and alcohol use",
            "Previous eye trauma or surgery",
            "Prolonged corticosteroid use",
        ],
        "management": (
            "Early cataracts may be managed with updated prescriptions and better lighting. "
            "When vision significantly impairs daily activities, phacoemulsification surgery "
            "with intraocular lens implant is the standard treatment."
        ),
        "prevalence": "~94 million visually impaired worldwide",
        "screening_freq": "Annual after age 60; earlier if symptomatic",
        "icd_code": "H26",
    },
    "diabetic_retinopathy": {
        "title": "🔴 Diabetic Retinopathy",
        "description": (
            "Diabetic retinopathy is a microvascular complication of diabetes that damages "
            "retinal blood vessels, potentially leading to retinal detachment and blindness. "
            "It is the leading cause of new blindness in working-age adults."
        ),
        "symptoms": [
            "Blurred or fluctuating vision",
            "Dark spots, strings, or floaters",
            "Difficulty seeing colours",
            "Dark or empty areas in vision",
            "Vision loss in advanced stages",
        ],
        "risk_factors": [
            "Duration of diabetes (highest risk after 20+ years)",
            "Poor glycaemic (HbA1c) control",
            "Hypertension",
            "Dyslipidaemia",
            "Nephropathy (kidney disease)",
            "Pregnancy in people with diabetes",
        ],
        "management": (
            "Requires systemic control (HbA1c, blood pressure, lipids) plus ophthalmologic "
            "intervention: anti-VEGF injections, laser photocoagulation, or vitrectomy."
        ),
        "prevalence": "~103 million worldwide",
        "screening_freq": "Annually for all diabetic patients",
        "icd_code": "E11.3",
    },
}


# =============================================================================
# FOLLOW-UP GUIDELINES
# Based on AAO Preferred Practice Patterns & ADA Standards of Care 2023
# =============================================================================

FOLLOWUP_GUIDELINES = {
    "glaucoma": {
        "high": {
            "urgency": "🔴 URGENT", "urgency_color": "#dc2626",
            "referral": "Refer to ophthalmology within 1–2 weeks",
            "next_screening": "Follow-up in 1–3 months after specialist review",
            "lifestyle": [
                "Avoid activities that raise intraocular pressure (heavy lifting, inverted yoga poses)",
                "Limit caffeine intake",
                "Sleep with head slightly elevated",
                "Do not skip prescribed eye drops",
                "Avoid tight collars or neck compression",
            ],
            "monitoring": [
                "Intraocular pressure measurement at every visit",
                "Visual field test every 6 months",
                "Optic disc photography annually",
                "OCT of retinal nerve fibre layer every 6–12 months",
            ],
            "source": "AAO Preferred Practice Pattern — Primary Open-Angle Glaucoma (2020)",
        },
        "medium": {
            "urgency": "🟡 MODERATE", "urgency_color": "#a16207",
            "referral": "Schedule ophthalmology appointment within 4–6 weeks",
            "next_screening": "Follow-up in 3–6 months",
            "lifestyle": [
                "Maintain healthy blood pressure",
                "Regular aerobic exercise may lower intraocular pressure",
                "Avoid smoking",
                "Wear UV-protective eyewear outdoors",
            ],
            "monitoring": [
                "IOP check at each visit",
                "Visual field test annually",
                "Optic disc assessment annually",
            ],
            "source": "AAO Preferred Practice Pattern — Primary Open-Angle Glaucoma (2020)",
        },
        "low": {
            "urgency": "🟢 ROUTINE", "urgency_color": "#15803d",
            "referral": "Routine ophthalmology review within 6 months",
            "next_screening": "Annual comprehensive eye exam",
            "lifestyle": [
                "Regular aerobic exercise",
                "Healthy diet rich in leafy greens",
                "UV eye protection outdoors",
            ],
            "monitoring": [
                "IOP at each annual visit",
                "Optic disc evaluation annually",
            ],
            "source": "AAO Preferred Practice Pattern — Primary Open-Angle Glaucoma (2020)",
        },
    },
    "diabetic_retinopathy": {
        "high": {
            "urgency": "🔴 URGENT", "urgency_color": "#dc2626",
            "referral": "Refer to retinal specialist within 1 week",
            "next_screening": "Re-evaluate in 1 month after specialist review",
            "lifestyle": [
                "Strict glycaemic control — target HbA1c < 7%",
                "Blood pressure control — target < 130/80 mmHg",
                "Low-fat, low-sugar diet",
                "No smoking — accelerates retinal vessel damage",
                "Avoid Valsalva manoeuvre (straining)",
            ],
            "monitoring": [
                "Dilated fundus exam every 3 months until stable",
                "HbA1c measurement every 3 months",
                "Blood pressure monitoring at every visit",
                "Fluorescein angiography as indicated by specialist",
                "OCT of macula to detect macular oedema",
            ],
            "source": "AAO Preferred Practice Pattern — Diabetic Retinopathy (2019); ADA Standards 2023",
        },
        "medium": {
            "urgency": "🟡 MODERATE", "urgency_color": "#a16207",
            "referral": "Schedule retinal specialist appointment within 2–4 weeks",
            "next_screening": "Follow-up in 3 months",
            "lifestyle": [
                "Optimise blood glucose — consult endocrinologist",
                "Control blood pressure and cholesterol",
                "Regular moderate exercise (with physician clearance)",
                "Annual kidney function tests (DR and nephropathy are linked)",
            ],
            "monitoring": [
                "Dilated fundus exam every 6 months",
                "HbA1c every 3 months",
                "Annual lipid panel",
            ],
            "source": "AAO Preferred Practice Pattern — Diabetic Retinopathy (2019)",
        },
        "low": {
            "urgency": "🟢 ROUTINE", "urgency_color": "#15803d",
            "referral": "Schedule ophthalmology appointment within 3 months",
            "next_screening": "Annual dilated fundus exam",
            "lifestyle": [
                "Maintain HbA1c < 7%",
                "Regular exercise and balanced diet",
                "Annual eye exams are mandatory for all diabetic patients",
            ],
            "monitoring": [
                "Annual dilated eye exam",
                "HbA1c every 3–6 months",
                "Blood pressure at each medical visit",
            ],
            "source": "ADA Standards of Medical Care in Diabetes 2023",
        },
    },
    "cataract": {
        "high": {
            "urgency": "🟡 ELECTIVE", "urgency_color": "#a16207",
            "referral": "Elective referral to ophthalmologist for surgical evaluation",
            "next_screening": "Surgical consultation within 4–8 weeks if vision significantly impaired",
            "lifestyle": [
                "Use bright lighting for reading and close work",
                "Use anti-glare coating on glasses",
                "Wear UV-blocking sunglasses outdoors",
                "Update eyeglass prescription to current level",
                "Avoid driving at night if glare is significant",
            ],
            "monitoring": [
                "Visual acuity check every 6 months",
                "Slit-lamp biomicroscopy to grade cataract",
                "Pre-surgical biometry (A-scan) before IOL implant surgery",
            ],
            "source": "AAO Preferred Practice Pattern — Cataract in the Adult Eye (2021)",
        },
        "medium": {
            "urgency": "🟢 ROUTINE", "urgency_color": "#15803d",
            "referral": "Schedule ophthalmology appointment within 3 months",
            "next_screening": "Re-evaluate in 6–12 months",
            "lifestyle": [
                "Wear sunglasses with UV400 protection",
                "Balanced diet with antioxidants (vitamins C and E)",
                "Control diabetes and blood pressure",
            ],
            "monitoring": [
                "Annual visual acuity testing",
                "Slit-lamp exam at each visit",
            ],
            "source": "AAO Preferred Practice Pattern — Cataract in the Adult Eye (2021)",
        },
        "low": {
            "urgency": "🟢 ROUTINE", "urgency_color": "#15803d",
            "referral": "Routine annual eye examination",
            "next_screening": "Annual review",
            "lifestyle": [
                "UV protection outdoors",
                "Antioxidant-rich diet",
                "Avoid smoking",
            ],
            "monitoring": [
                "Annual visual acuity and lens assessment",
            ],
            "source": "AAO Preferred Practice Pattern — Cataract in the Adult Eye (2021)",
        },
    },
    "normal": {
        "high": {
            "urgency": "🟢 ROUTINE", "urgency_color": "#15803d",
            "referral": "Continue routine annual eye examinations",
            "next_screening": "Annual comprehensive eye exam",
            "lifestyle": [
                "Balanced diet rich in leafy greens and omega-3 fatty acids",
                "UV-protective eyewear outdoors",
                "Regular exercise to maintain healthy blood pressure",
                "Avoid smoking",
            ],
            "monitoring": [
                "Annual dilated eye exam",
                "IOP check if family history of glaucoma",
                "Blood pressure and glucose monitoring",
            ],
            "source": "AAO Comprehensive Adult Medical Eye Evaluation Guidelines (2020)",
        },
        "medium": {
            "urgency": "🟢 ROUTINE", "urgency_color": "#15803d",
            "referral": "Continue routine annual eye examinations",
            "next_screening": "Annual comprehensive eye exam",
            "lifestyle": [
                "Maintain healthy lifestyle",
                "UV-protective eyewear outdoors",
                "Regular exercise and balanced diet",
            ],
            "monitoring": ["Annual comprehensive eye exam"],
            "source": "AAO Comprehensive Adult Medical Eye Evaluation Guidelines (2020)",
        },
        "low": {
            "urgency": "🟢 ROUTINE", "urgency_color": "#15803d",
            "referral": "Continue routine annual eye examinations",
            "next_screening": "Annual comprehensive eye exam",
            "lifestyle": [
                "Maintain healthy lifestyle",
                "UV-protective eyewear outdoors",
                "Regular exercise and balanced diet",
            ],
            "monitoring": ["Annual comprehensive eye exam"],
            "source": "AAO Comprehensive Adult Medical Eye Evaluation Guidelines (2020)",
        },
    },
}


# =============================================================================
# MODEL LOADING
# =============================================================================

@st.cache_resource
def load_cnn_model():
    try:
        loaded_model = keras.models.load_model(MODEL_PATH, compile=False)
        _ = loaded_model(
            tf.zeros((1, 224, 224, 3), dtype=tf.float32), training=False)
        return loaded_model, None
    except Exception as exc:
        return None, str(exc)


model, model_error = load_cnn_model()

if model is None:
    st.error(f"❌ Model loading failed:\n\n{model_error}")
    st.info("Ensure the Hugging Face repo contains 'best_model.h5'.")
    st.stop()


# =============================================================================
# PREPROCESS
# =============================================================================

def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB")
    original = img.copy()
    resized = img.resize((224, 224), Image.Resampling.LANCZOS)
    arr = np.array(resized, dtype=np.float32) / 255.0
    batch = np.expand_dims(arr, axis=0)
    return batch, original, resized


# =============================================================================
# IMAGE QUALITY ASSESSMENT
# =============================================================================

def calculate_image_quality(image):
    rgb = image.convert("RGB")
    w, h = rgb.size
    arr = np.asarray(rgb, dtype=np.float32)
    gray = np.mean(arr, axis=2)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    lap = ndimage.laplace(gray)
    sharpness = float(np.var(lap))

    smooth = ndimage.gaussian_filter(gray, sigma=1.0)
    noise = float(np.std(gray - smooth))
    snr = brightness / (noise + 1e-6)

    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 255))
    prob = hist / (hist.sum() + 1e-9)
    entropy = float(-np.sum(prob[prob > 0] * np.log2(prob[prob > 0])))

    total_px = w * h

    def score_band(v, lo1, hi1, lo2, hi2, s_good, s_ok, s_bad):
        if lo1 <= v <= hi1:
            return s_good
        if lo2 <= v <= hi2:
            return s_ok
        return s_bad

    res_score = 100 if total_px >= 1_000_000 else 90 if total_px >= 500_000 else 70 if total_px >= 200_000 else 40
    bri_score = score_band(brightness, 45, 210, 30, 230, 100, 70, 35)
    con_score = 100 if contrast >= 45 else 70 if contrast >= 25 else 40
    sha_score = 100 if sharpness >= 150 else 70 if sharpness >= 50 else 35
    snr_score = 100 if snr >= 10 else 70 if snr >= 5 else 40

    is_rgb = rgb.mode == "RGB"
    ch_score = 100 if is_rgb else 80

    overall = float(np.clip(
        res_score*0.18 + bri_score*0.18 + con_score*0.18 +
        sha_score*0.25 + snr_score*0.12 + ch_score*0.09,
        0, 100
    ))

    def status(score, labels):
        if score >= 85:
            return labels[0]
        if score >= 65:
            return labels[1]
        return labels[2]

    warnings = []
    if total_px < 200_000:
        warnings.append("Low resolution — prediction reliability may be reduced.")
    if brightness < 30:
        warnings.append("Image appears very dark.")
    elif brightness > 230:
        warnings.append("Image appears overexposed.")
    if contrast < 25:
        warnings.append("Low contrast detected.")
    if sharpness < 50:
        warnings.append("Image may be blurry or out of focus.")
    if snr < 5:
        warnings.append("High noise level detected.")
    if not is_rgb:
        warnings.append("Image was converted to RGB before analysis.")

    return dict(
        width=w, height=h, total_px=total_px,
        brightness=brightness, brightness_status=status(
            bri_score, ["🟢 Good", "🟡 Moderate", "🔴 Extreme"]),
        contrast=contrast,   contrast_status=status(
            con_score,  ["🟢 Good", "🟡 Moderate", "🔴 Low"]),
        sharpness=sharpness, sharpness_status=status(
            sha_score, ["🟢 Sharp", "🟡 Moderate", "🔴 Blurry"]),
        snr=snr,             snr_status=status(
            snr_score,       ["🟢 Good", "🟡 Moderate", "🔴 Noisy"]),
        entropy=entropy,
        resolution_status=status(res_score, [
            "🟢 Excellent", "🟢 Good", "🟡 Acceptable" if res_score == 70 else "🔴 Low"]),
        channel_status="🟢 RGB" if is_rgb else "🟡 Converted to RGB",
        overall_score=overall,
        overall_status=status(overall, ["🟢 Good", "🟡 Moderate", "🔴 Poor"]),
        warnings=warnings,
    )


def display_image_quality_assessment(image, title="Image Quality Assessment"):
    q = calculate_image_quality(image)
    st.subheader(title)
    st.caption("Technical image metrics only — do not determine clinical validity.")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Overall Score", f"{q['overall_score']:.1f}/100")
    with c2:
        score = q['overall_score']
        if score >= 85:
            st.success(q["overall_status"])
        elif score >= 65:
            st.warning(q["overall_status"])
        else:
            st.error(q["overall_status"])

    st.progress(q['overall_score'] / 100)

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Resolution", f"{q['width']}×{q['height']}")
        st.caption(q["resolution_status"])
    with r2:
        st.metric("Brightness", f"{q['brightness']:.1f}")
        st.caption(q["brightness_status"])
    with r3:
        st.metric("Contrast", f"{q['contrast']:.1f}")
        st.caption(q["contrast_status"])
    with r4:
        st.metric("Sharpness", f"{q['sharpness']:.1f}")
        st.caption(q["sharpness_status"])

    r5, r6 = st.columns(2)
    with r5:
        st.metric("SNR", f"{q['snr']:.1f}")
        st.caption(q["snr_status"])
    with r6:
        st.metric("Entropy", f"{q['entropy']:.2f} bits")
        st.caption("Higher = richer texture")

    if q["warnings"]:
        for w in q["warnings"]:
            st.warning(f"⚠️ {w}")
    else:
        st.success("✅ No major quality issues detected.")

    return q


# =============================================================================
# PREDICT
# =============================================================================

def predict_image(image_file):
    batch, original, resized = preprocess_image(image_file)
    preds = model(batch, training=False)
    probs = preds.numpy()[0]
    idx = int(np.argmax(probs))
    return idx, probs, batch, original, resized


# =============================================================================
# GRAD-CAM
# =============================================================================

def get_gradcam_layers():
    layers = []
    for layer in model.layers:
        try:
            if (len(layer.output.shape) == 4 and
                    isinstance(layer, keras.layers.Conv2D)):
                layers.append(layer.name)
        except Exception:
            continue
    return layers


GRADCAM_LAYERS = get_gradcam_layers()
GRADCAM_DEFAULT = GRADCAM_LAYERS[-1] if GRADCAM_LAYERS else ""


@st.cache_resource
def build_gradcam_model(layer_name):
    try:
        target_idx = next(
            i for i, l in enumerate(model.layers) if l.name == layer_name
        )
    except StopIteration:
        return None, f"Layer '{layer_name}' not found."

    inp = keras.Input(shape=(224, 224, 3), name=f"gc_in_{layer_name}")
    x = inp
    conv_out = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == layer_name:
            conv_out = x

    if conv_out is None:
        return None, "Could not capture conv output."

    try:
        grad_model = keras.Model(
            inputs=inp,
            outputs=[conv_out, x],
            name=f"gc_{layer_name}",
        )
        return grad_model, None
    except Exception as exc:
        return None, str(exc)


def generate_gradcam_heatmap(image_batch, class_index, layer_name):
    grad_model, err = build_gradcam_model(layer_name)
    if grad_model is None:
        raise RuntimeError(f"Grad-CAM model error: {err}")

    img_tensor = tf.cast(image_batch, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        class_score = predictions[:, class_index]

    grads = tape.gradient(class_score, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradient is None — check layer connectivity.")

    pooled_grads = tf.reduce_mean(grads, axis=[0, 1, 2])
    conv_map = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_map * pooled_grads, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    hmax = tf.reduce_max(heatmap)
    if hmax == 0:
        heatmap = tf.zeros_like(heatmap)
    else:
        heatmap = heatmap / hmax

    return heatmap.numpy()


def apply_heatmap_to_image(original_image, heatmap, colormap="jet", alpha=0.45):
    orig_rgb = original_image.convert("RGB")
    W, H = orig_rgb.size

    heat_img = Image.fromarray(np.uint8(heatmap * 255), mode="L")
    heat_img = heat_img.resize((W, H), Image.Resampling.BILINEAR)
    heat_np = np.array(heat_img) / 255.0

    cmap = plt.get_cmap(colormap)
    colored = np.uint8(cmap(heat_np)[:, :, :3] * 255)
    colored_pil = Image.fromarray(colored).convert("RGB")

    overlay = Image.blend(orig_rgb, colored_pil, alpha=alpha)
    return colored_pil, overlay


# =============================================================================
# PROBABILITY BAR CHART
# =============================================================================

def plot_probability_bars(probs, title="Class Probabilities"):
    labels = [DISPLAY_NAMES[c] for c in class_names]
    colors = [CLASS_COLORS_MPL[c] for c in class_names]
    values = [probs[i] * 100 for i in range(len(class_names))]

    fig, ax = plt.subplots(figsize=(5, 2.6))
    bars = ax.barh(labels, values, color=colors,
                   height=0.5, edgecolor="none", alpha=0.88)
    for bar, val in zip(bars, values):
        ax.text(
            min(val + 1.2, 96), bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", ha="left",
            color="#1e293b", fontsize=8.5, fontweight="600",
        )
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)", fontsize=8, color="#475569")
    ax.set_title(title, fontsize=9, fontweight="600", pad=6, color="#0f172a")
    ax.tick_params(labelsize=8.5, colors="#374151")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.spines["left"].set_color("#cbd5e1")
    ax.axvline(50, color="#e2e8f0", linewidth=0.8, linestyle="--")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plt.tight_layout(pad=0.4)
    return fig


# =============================================================================
# RISK STRATIFICATION
# =============================================================================

def compute_risk_profile(class_name, confidence, quality_score):
    base_risk, action = CLASS_RISK[class_name]
    eff_conf = confidence
    if quality_score < 65:
        eff_conf *= 0.75
        q_note = "⚠️ Poor image quality — prediction reliability is reduced."
    elif quality_score < 85:
        eff_conf *= 0.90
        q_note = "Image quality is moderate."
    else:
        q_note = "Image quality is good."

    if eff_conf >= 85:
        conf_tier = "High Confidence"
    elif eff_conf >= 60:
        conf_tier = "Moderate Confidence"
    else:
        conf_tier = "Low Confidence"

    return dict(
        base_risk=base_risk, action=action,
        eff_conf=eff_conf, conf_tier=conf_tier, q_note=q_note,
    )


# =============================================================================
# FOLLOW-UP SCHEDULING FUNCTIONS
# =============================================================================

def generate_followup_recommendation(class_name, confidence, quality_score):
    """
    Derives effective confidence (adjusted for image quality),
    maps it to a tier (high/medium/low), and returns the
    corresponding guideline-based recommendation dict.
    """
    eff_conf = confidence
    if quality_score < 65:
        eff_conf *= 0.75
    elif quality_score < 85:
        eff_conf *= 0.90

    # Normal always gets routine regardless of confidence
    if class_name == "normal":
        tier = "low"
    elif eff_conf >= 80:
        tier = "high"
    elif eff_conf >= 55:
        tier = "medium"
    else:
        tier = "low"

    g = FOLLOWUP_GUIDELINES[class_name][tier]
    return dict(
        class_name=class_name,
        confidence=confidence,
        eff_conf=eff_conf,
        tier=tier,
        urgency=g["urgency"],
        urgency_color=g["urgency_color"],
        referral=g["referral"],
        next_screening=g["next_screening"],
        lifestyle=g["lifestyle"],
        monitoring=g["monitoring"],
        source=g["source"],
    )


def display_followup_card(fu, eye_label):
    """Render a structured follow-up recommendation card."""
    cls = fu["class_name"]
    col = CLASS_COLORS_MPL[cls]
    urg = fu["urgency_color"]

    lifestyle_html = "".join(
        f'<div class="fu-row">'
        f'<div class="fu-bullet" style="background:{col};"></div>'
        f'<div class="fu-text">{item}</div></div>'
        for item in fu["lifestyle"]
    )
    monitoring_html = "".join(
        f'<div class="fu-row">'
        f'<div class="fu-bullet" style="background:#64748b;"></div>'
        f'<div class="fu-text">{item}</div></div>'
        for item in fu["monitoring"]
    )

    st.markdown(f"""
    <div class="fu-card">
      <div class="fu-title">📅 {eye_label} — Follow-Up Recommendation</div>

      <div style="margin-bottom:.8rem;">
        <span style="background:{urg}22;color:{urg};border:1px solid {urg}55;
                     padding:4px 14px;border-radius:20px;font-size:.82rem;font-weight:700;">
          {fu['urgency']}
        </span>&nbsp;
        <span style="font-size:.82rem;color:#475569;">
          Effective confidence: <strong>{fu['eff_conf']:.1f}%</strong>
          (Tier: <strong>{fu['tier'].upper()}</strong>)
        </span>
      </div>

      <div style="margin:.5rem 0;">
        <span style="font-size:.8rem;font-weight:700;color:#1e40af;
                     text-transform:uppercase;letter-spacing:.6px;">Referral</span>
        <div class="fu-text" style="margin-top:2px;">🏥 {fu['referral']}</div>
      </div>

      <div style="margin:.5rem 0;">
        <span style="font-size:.8rem;font-weight:700;color:#1e40af;
                     text-transform:uppercase;letter-spacing:.6px;">Next Screening</span>
        <div class="fu-text" style="margin-top:2px;">🗓️ {fu['next_screening']}</div>
      </div>

      <div style="margin:.6rem 0 .3rem;">
        <span style="font-size:.8rem;font-weight:700;color:#1e40af;
                     text-transform:uppercase;letter-spacing:.6px;">Lifestyle Recommendations</span>
        <div style="margin-top:4px;">{lifestyle_html}</div>
      </div>

      <div style="margin:.6rem 0 .3rem;">
        <span style="font-size:.8rem;font-weight:700;color:#1e40af;
                     text-transform:uppercase;letter-spacing:.6px;">Monitoring Parameters</span>
        <div style="margin-top:4px;">{monitoring_html}</div>
      </div>

      <div style="margin-top:.8rem;padding-top:.6rem;border-top:1px solid #e2e8f0;">
        <span style="font-size:.72rem;color:#94a3b8;">
          📖 Guideline source: {fu['source']}
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# COLORMAP DICT
# =============================================================================

GRAD_COLORMAPS = {
    "Jet (Classic)":     "jet",
    "Hot (Thermal)":     "hot",
    "Inferno":           "inferno",
    "Plasma":            "plasma",
    "Viridis":           "viridis",
    "RdYlGn (Traffic)": "RdYlGn",
}


# =============================================================================
# PREPROCESSING PREVIEW
# =============================================================================

def preprocessing_views(image):
    rgb = image.convert("RGB")
    resized = rgb.resize((224, 224), Image.Resampling.LANCZOS)
    gray = ImageOps.grayscale(resized)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    enhanced = ImageEnhance.Contrast(resized).enhance(2.0)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.5)
    return {
        "Original (224×224)": resized,
        "Contrast Enhanced":  enhanced,
        "Grayscale":          gray.convert("RGB"),
        "Edge Detection":     edges.convert("RGB"),
    }


# =============================================================================
# ATTENTION FOCUS
# =============================================================================

def attention_focus(heatmap):
    flat = np.sort(heatmap.flatten())
    n = len(flat)
    cum = np.cumsum(flat)
    tot = cum[-1] + 1e-9
    gini = float(np.clip((n + 1 - 2 * np.sum(cum) / tot) / n, 0, 1))
    if gini > 0.6:
        label = "🎯 Highly Focused"
        desc = "Attention is concentrated in a specific region."
    elif gini > 0.35:
        label = "📍 Moderately Focused"
        desc = "Attention spans a moderate area."
    else:
        label = "🌐 Diffuse Attention"
        desc = "Attention is spread across the image."
    return dict(score=gini, label=label, desc=desc)


# =============================================================================
# BILATERAL SYMMETRY
# =============================================================================

def bilateral_symmetry(lp, rp, l_idx, r_idx):
    diff = np.abs(lp - rp)
    kl = float(np.sum(lp * np.log((lp + 1e-9) / (rp + 1e-9))))
    agree = l_idx == r_idx
    sym_score = float((1.0 - np.mean(diff)) * 100)
    if agree and sym_score >= 85:
        label = "🟢 High Bilateral Agreement"
        desc = "Both eyes show consistent predictions."
        risk = "low"
    elif agree:
        label = "🟡 Moderate Bilateral Agreement"
        desc = "Same class but confidence levels differ."
        risk = "medium"
    else:
        label = "🔴 Asymmetric Prediction"
        desc = "Different classifications per eye — may be clinically significant."
        risk = "high"
    return dict(
        agree=agree, sym_score=sym_score, kl=kl,
        label=label, desc=desc, risk=risk, diff=diff,
    )


# =============================================================================
# ACTIVATION DISTRIBUTION
# =============================================================================

def plot_activation_distribution(heatmap, eye_label):
    flat = heatmap.flatten()
    fig, ax = plt.subplots(figsize=(5, 2.4))
    ax.hist(flat, bins=40, color="#3b82f6", alpha=0.75, edgecolor="none")
    ax.axvline(float(np.mean(flat)), color="#ef4444", lw=1.5, ls="--",
               label=f"Mean {np.mean(flat):.2f}")
    ax.axvline(float(np.percentile(flat, 90)), color="#f59e0b", lw=1.5, ls=":",
               label=f"P90 {np.percentile(flat, 90):.2f}")
    ax.set_title(f"Activation Distribution — {eye_label}", fontsize=9, fontweight="600")
    ax.set_xlabel("Activation", fontsize=8, color="#475569")
    ax.set_ylabel("Count", fontsize=8, color="#475569")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7.5)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plt.tight_layout(pad=0.4)
    return fig


# =============================================================================
# COUNTERFACTUAL
# =============================================================================

def counterfactual(probs, top_idx):
    order = np.argsort(probs)[::-1]
    second = order[1]
    margin = float((probs[top_idx] - probs[second]) * 100)
    s_name = DISPLAY_NAMES[class_names[second]]
    s_conf = float(probs[second] * 100)
    if margin > 40:
        stab = "🟢 Very Stable"
        desc = "Model is highly decisive."
    elif margin > 15:
        stab = "🟡 Moderately Stable"
        desc = f"Gap to '{s_name}' is moderate."
    else:
        stab = "🔴 Uncertain"
        desc = f"'{s_name}' ({s_conf:.1f}%) is a close competitor."
    return dict(margin=margin, second_name=s_name, second_conf=s_conf,
                stability=stab, desc=desc)


# =============================================================================
# RGB HISTOGRAMS
# =============================================================================

def plot_rgb_histograms(image, title="RGB Channels"):
    arr = np.array(image.convert("RGB"))
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.4), sharey=True)
    ch_names = ["Red", "Green", "Blue"]
    ch_colors = ["#ef4444", "#22c55e", "#3b82f6"]
    for ax, name, col, i in zip(axes, ch_names, ch_colors, range(3)):
        ax.hist(arr[:, :, i].flatten(), bins=60,
                color=col, alpha=0.75, edgecolor="none")
        ax.set_title(name, fontsize=9, fontweight="600")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Intensity", fontsize=7.5, color="#475569")
        ax.set_facecolor("white")
    fig.suptitle(title, fontsize=10, fontweight="600", y=1.02)
    fig.patch.set_facecolor("white")
    plt.tight_layout(pad=0.4)
    return fig


# =============================================================================
# SESSION HISTORY
# =============================================================================

def init_history():
    if "history" not in st.session_state:
        st.session_state["history"] = []


def add_history(lc, lconf, rc, rconf):
    init_history()
    st.session_state["history"].append(dict(
        ts=datetime.datetime.now().strftime("%H:%M:%S"),
        left=DISPLAY_NAMES[lc],  lconf=lconf,
        right=DISPLAY_NAMES[rc], rconf=rconf,
        agree=lc == rc,
    ))


# =============================================================================
# REPORTS
# =============================================================================

def text_report(lci, lp, rci, rp):
    lc, rc = class_names[lci], class_names[rci]
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 60,
        "OCULAR DISEASE AI CLASSIFICATION REPORT",
        "=" * 60,
        f"Generated : {now}",
        "Disclaimer: Educational use only. NOT a medical diagnosis.",
        "",
        "-" * 60, "LEFT EYE", "-" * 60,
        f"Prediction       : {DISPLAY_NAMES[lc]}",
        f"Confidence       : {lp[lci]*100:.2f}%",
        f"Risk Level       : {CLASS_RISK[lc][0].upper()}",
        f"Recommended Actn : {CLASS_RISK[lc][1]}",
        "", "All probabilities:",
    ]
    for i, p in enumerate(lp):
        marker = " ◀" if i == lci else ""
        lines.append(f"  {DISPLAY_NAMES[class_names[i]]:<28} {p*100:.2f}%{marker}")
    lines += [
        "", "-" * 60, "RIGHT EYE", "-" * 60,
        f"Prediction       : {DISPLAY_NAMES[rc]}",
        f"Confidence       : {rp[rci]*100:.2f}%",
        f"Risk Level       : {CLASS_RISK[rc][0].upper()}",
        f"Recommended Actn : {CLASS_RISK[rc][1]}",
        "", "All probabilities:",
    ]
    for i, p in enumerate(rp):
        marker = " ◀" if i == rci else ""
        lines.append(f"  {DISPLAY_NAMES[class_names[i]]:<28} {p*100:.2f}%{marker}")
    lines += [
        "", "-" * 60, "BILATERAL SUMMARY", "-" * 60,
        f"Agreement : {'Yes' if lc == rc else 'No — asymmetric findings'}",
        f"Left      : {DISPLAY_NAMES[lc]} @ {lp[lci]*100:.2f}%",
        f"Right     : {DISPLAY_NAMES[rc]} @ {rp[rci]*100:.2f}%",
        "", "-" * 60, "MODEL INFO", "-" * 60,
        "Architecture : Custom CNN",
        "Input size   : 224 × 224 × 3",
        f"Conv layers  : {len(GRADCAM_LAYERS)}",
        "", "=" * 60, "END OF REPORT", "=" * 60,
    ]
    return "\n".join(lines)


def json_report(lci, lp, rci, rp, lq=None, rq=None):
    lc, rc = class_names[lci], class_names[rci]
    data = dict(
        metadata=dict(
            type="Ocular Disease AI Classification",
            generated=datetime.datetime.now().isoformat(),
            model="Custom CNN", input_size="224x224x3",
            disclaimer="Educational use only. NOT a medical diagnosis.",
        ),
        left_eye=dict(
            predicted=DISPLAY_NAMES[lc],
            confidence_pct=round(float(lp[lci]) * 100, 2),
            risk=CLASS_RISK[lc][0],
            all_probs={DISPLAY_NAMES[class_names[i]]: round(
                float(p) * 100, 4) for i, p in enumerate(lp)},
            image_quality_score=round(lq["overall_score"], 2) if lq else None,
        ),
        right_eye=dict(
            predicted=DISPLAY_NAMES[rc],
            confidence_pct=round(float(rp[rci]) * 100, 2),
            risk=CLASS_RISK[rc][0],
            all_probs={DISPLAY_NAMES[class_names[i]]: round(
                float(p) * 100, 4) for i, p in enumerate(rp)},
            image_quality_score=round(rq["overall_score"], 2) if rq else None,
        ),
        bilateral=dict(
            agreement=lc == rc,
            recommended_action=(
                CLASS_RISK[lc][1] if lc == rc
                else "Clinical review required — asymmetric findings."
            ),
        ),
    )
    return json.dumps(data, indent=2)


def pil_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# =============================================================================
# INIT
# =============================================================================

init_history()
if "sel_disease" not in st.session_state:
    st.session_state["sel_disease"] = "glaucoma"


# =============================================================================
# HERO BANNER
# =============================================================================

st.markdown("""
<div class="hero-wrap">
  <div class="hero-badge">AI-Powered Ophthalmology</div>
  <div class="hero-title">👁️ Ocular Disease Detection</div>
  <div class="hero-sub">Deep learning retinal image analysis · Grad-CAM explainability · Risk stratification · Clinical Follow-Up Scheduling</div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## 👁️ Ocular Disease Detection")
    st.markdown('<div class="sec-title">Model Info</div>', unsafe_allow_html=True)
    st.markdown(f"""
- **Architecture:** Custom CNN
- **Input:** 224 × 224 px
- **Output:** 4 classes
- **XAI:** Grad-CAM
- **Conv layers:** {len(GRADCAM_LAYERS)}
""")

    st.markdown("---")
    st.markdown('<div class="sec-title">Disease Legend</div>', unsafe_allow_html=True)
    risk_colors = {"low": "#15803d", "medium": "#a16207", "high": "#dc2626"}
    risk_bgs    = {"low": "#dcfce7", "medium": "#fef9c3", "high": "#fee2e2"}
    for k in class_names:
        rc = risk_colors[CLASS_RISK[k][0]]
        bg = risk_bgs[CLASS_RISK[k][0]]
        st.markdown(
            f'<div style="margin:5px 0;padding:7px 10px;background:#fff;'
            f'border-radius:8px;border-left:3px solid {rc};">'
            f'<strong>{CLASS_ICONS[k]} {DISPLAY_NAMES[k]}</strong><br>'
            f'<span style="background:{bg};color:{rc};font-size:.72rem;'
            f'font-weight:700;padding:1px 8px;border-radius:10px;">'
            f'{CLASS_RISK[k][0].upper()} RISK</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    hist = st.session_state.get("history", [])
    if hist:
        st.markdown(f'<div class="sec-title">Session History ({len(hist)})</div>',
                    unsafe_allow_html=True)
        for e in reversed(hist[-3:]):
            agree_sym = "✅" if e["agree"] else "⚠️"
            st.markdown(
                f'<div style="font-size:.78rem;padding:3px 0;color:#475569;">'
                f'<span style="color:#3b82f6;font-weight:600;">{e["ts"]}</span><br>'
                f'{agree_sym} L:{e["left"]} · R:{e["right"]}</div>',
                unsafe_allow_html=True,
            )
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state["history"] = []
            st.rerun()

    st.markdown("---")
    st.caption("⚠️ Not a medical device. Educational use only.")


# =============================================================================
# MAIN TABS  — About is first, Follow-Up added after Report
# =============================================================================

(tab_about, tab_pred, tab_xai, tab_analysis,
 tab_bilateral, tab_report, tab_followup,
 tab_viz, tab_disease, tab_how) = st.tabs([
    "📖 About",
    "🔍 Predict",
    "🔬 Explainable AI",
    "📊 Analysis",
    "⚖️ Bilateral",
    "📋 Report",
    "📅 Follow-Up",
    "📈 Visualizations",
    "📚 Disease Library",
    "⚙️ How It Works",
])


# =============================================================================
# TAB — ABOUT  (now first)
# =============================================================================

with tab_about:
    st.header("📖 About This Project")

    if os.path.isfile(HERO_IMAGE_PATH):
        st.image(HERO_IMAGE_PATH, use_container_width=True)

    st.markdown("""
**Ocular Disease Detection** is a deep learning classification system for retinal
image analysis. It demonstrates how modern CNNs and explainability techniques
can be combined in a clinician-friendly interface for educational and research purposes.

### 🎯 Project Goals

| Goal | Details |
|---|---|
| Automated Screening | Classify 4 major ocular diseases from retinal photography |
| Explainability | Multi-layer, multi-colormap Grad-CAM with focus scoring |
| Confidence Calibration | Quality assessment + counterfactual stability analysis |
| Bilateral Context | Inter-eye symmetry scoring mirrors real clinical workflows |
| Clinical Follow-Up | Guideline-based scheduling recommendations per prediction |
| Accessibility | Browser-based, no local installation required |

### ✨ Features
""")

    feats = [
        "Advanced image quality assessment — SNR, entropy, colour balance",
        "Probability bar charts for all 4 classes per eye",
        "Risk stratification engine (Low / Medium / High) with clinical action guidance",
        "Multi-colormap Grad-CAM — Jet, Hot, Inferno, Plasma, Viridis, RdYlGn",
        "Preprocessing pipeline preview — 4 transformation views",
        "Attention focus score — Gini-based heatmap concentration metric",
        "Bilateral symmetry analysis — KL-divergence + per-class delta chart",
        "Enhanced disease library — ICD codes, global prevalence, screening frequency",
        "Heatmap activation distribution plots",
        "Counterfactual prediction stability analysis",
        "RGB channel histograms for retinal image colour analysis",
        "Session history tracker in sidebar",
        "Dual-format report download — structured .txt and .json",
        "Downloadable Grad-CAM overlay images",
        "Clean white UI with custom CSS cards, badges, and tabs",
        "📅 Follow-Up Scheduling — AAO/ADA guideline-based referral, screening, lifestyle & monitoring recommendations",
    ]
    for i, f in enumerate(feats, 1):
        st.markdown(f"**{i:02d}.** {f}")


# =============================================================================
# TAB — PREDICT
# =============================================================================

with tab_pred:
    st.header("🔍 Upload Retinal Images for Classification")
    st.markdown("Upload fundus photographs for both eyes. Accepted: **JPG, JPEG, PNG**.")

    ul, ur = st.columns(2)
    with ul:
        Left_Eye = st.file_uploader(
            "👁️ **Left Eye**", type=["jpg", "jpeg", "png"], key="left_up"
        )
    with ur:
        Right_Eye = st.file_uploader(
            "👁️ **Right Eye**", type=["jpg", "jpeg", "png"], key="right_up"
        )

    if Left_Eye or Right_Eye:
        st.markdown('<hr class="fdiv">', unsafe_allow_html=True)
        st.subheader("🧪 Preprocessing Pipeline Preview")
        st.caption(
            "Shows how the image is transformed before entering the model — "
            "original, contrast-enhanced, grayscale, and edge-detected."
        )
        src_file  = Left_Eye if Left_Eye else Right_Eye
        src_img   = Image.open(src_file).convert("RGB")
        views     = preprocessing_views(src_img)
        pc        = st.columns(4)
        for col, (name, img) in zip(pc, views.items()):
            with col:
                st.image(img, caption=name, use_container_width=True)

    st.markdown('<hr class="fdiv">', unsafe_allow_html=True)

    dl, dr = st.columns(2)
    with dl:
        if Left_Eye:
            st.image(Left_Eye, caption="Left Eye — Uploaded", use_container_width=True)
            lqi = Image.open(Left_Eye).convert("RGB")
            with st.expander("🖼️ Left Eye — Image Quality", expanded=True):
                left_q_data = display_image_quality_assessment(lqi, "Left Eye Quality")
    with dr:
        if Right_Eye:
            st.image(Right_Eye, caption="Right Eye — Uploaded", use_container_width=True)
            rqi = Image.open(Right_Eye).convert("RGB")
            with st.expander("🖼️ Right Eye — Image Quality", expanded=True):
                right_q_data = display_image_quality_assessment(rqi, "Right Eye Quality")

    if Left_Eye and Right_Eye:
        try:
            (lci, lp, lb, lo, lpr) = predict_image(Left_Eye)
            (rci, rp, rb, ro, rpr) = predict_image(Right_Eye)

            st.session_state.update(dict(
                left_prediction=(lci, lp, lb, lo, lpr),
                right_prediction=(rci, rp, rb, ro, rpr),
                left_quality=calculate_image_quality(lo),
                right_quality=calculate_image_quality(ro),
            ))

            add_history(class_names[lci], lp[lci]*100,
                        class_names[rci], rp[rci]*100)

            st.markdown('<hr class="fdiv">', unsafe_allow_html=True)
            st.subheader("🎯 Classification Results")

            rl, rr = st.columns(2)
            for col, cls_idx, probs, eye_label in [
                (rl, lci, lp, "Left Eye"),
                (rr, rci, rp, "Right Eye"),
            ]:
                with col:
                    cls    = class_names[cls_idx]
                    conf   = probs[cls_idx] * 100
                    is_dis = cls != "normal"
                    card_c = "rcard-disease" if is_dis else "rcard-normal"
                    bar_c  = "cbar-red"      if is_dis else "cbar-grn"
                    qq     = st.session_state["left_quality" if eye_label == "Left Eye" else "right_quality"]
                    rp_    = compute_risk_profile(cls, conf, qq["overall_score"])
                    bdg    = {"low": "badge-low", "medium": "badge-med", "high": "badge-high"}[rp_["base_risk"]]

                    st.markdown(f"""
                    <div class="rcard {card_c}">
                      <div style="font-size:.75rem;font-weight:700;
                                  color:{CLASS_COLORS_MPL[cls]};
                                  text-transform:uppercase;letter-spacing:1px;
                                  margin-bottom:4px;">
                        {CLASS_ICONS[cls]} {eye_label}
                      </div>
                      <div class="rcard-title">{DISPLAY_NAMES[cls]}</div>
                      <div class="rcard-conf">Confidence: <strong>{conf:.2f}%</strong></div>
                      <div class="cbar-bg">
                        <div class="{bar_c} cbar-fill" style="width:{conf:.1f}%"></div>
                      </div>
                      <div style="margin-top:10px;">
                        <span class="badge {bdg}">{rp_['base_risk'].upper()} RISK</span>
                        <div style="font-size:.78rem;color:#475569;margin-top:5px;">
                          {rp_['action']}
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    fig = plot_probability_bars(probs, f"{eye_label} — All Classes")
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    with st.expander("🔄 Prediction Stability"):
                        cf = counterfactual(probs, cls_idx)
                        st.markdown(f"**{cf['stability']}** — margin: `{cf['margin']:.1f}%`")
                        st.caption(cf['desc'])
                        st.markdown(f"Runner-up: **{cf['second_name']}** at `{cf['second_conf']:.1f}%`")

        except Exception as exc:
            st.error(f"Prediction error: {exc}")


# =============================================================================
# TAB — EXPLAINABLE AI
# =============================================================================

with tab_xai:
    st.header("🔬 Explainable AI — Multi-Layer Grad-CAM Lab")
    st.markdown(
        "**Grad-CAM** (Gradient-weighted Class Activation Mapping) highlights which "
        "spatial regions of the retinal image drove the model's classification decision."
    )
    st.warning(
        "⚠️ Grad-CAM heatmaps represent model attention, not anatomical disease locations. "
        "They are interpretability visualisations, not clinical localisations."
    )

    if not GRADCAM_LAYERS:
        st.error("No compatible Conv2D layers found for Grad-CAM.")
    elif "left_prediction" not in st.session_state:
        st.info("👈 Upload and classify both eyes in the **🔍 Predict** tab first.")
    else:
        (lci, lp, lb, lo, _) = st.session_state["left_prediction"]
        (rci, rp, rb, ro, _) = st.session_state["right_prediction"]

        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            sel_layer = st.selectbox(
                "🧠 Convolutional Layer",
                options=GRADCAM_LAYERS,
                index=len(GRADCAM_LAYERS) - 1,
                key="gc_layer",
            )
        with gc2:
            sel_cmap_label = st.selectbox(
                "🎨 Heatmap Colormap", list(GRAD_COLORMAPS.keys()), key="gc_cmap"
            )
            sel_cmap = GRAD_COLORMAPS[sel_cmap_label]
        with gc3:
            gc_alpha = st.slider("Overlay Intensity", 0.2, 0.8, 0.45, 0.05, key="gc_alpha")

        lobj = next((l for l in model.layers if l.name == sel_layer), None)
        if lobj:
            i1, i2, i3 = st.columns(3)
            with i1:
                st.metric("Layer", sel_layer)
            with i2:
                try:
                    s = lobj.output.shape
                    st.metric("Feature Map", f"{s[1]}×{s[2]}")
                except Exception:
                    st.metric("Feature Map", "N/A")
            with i3:
                try:
                    st.metric("Channels", str(lobj.output.shape[-1]))
                except Exception:
                    st.metric("Channels", "N/A")

        st.markdown('<hr class="fdiv">', unsafe_allow_html=True)

        for eye_label, cls_idx, probs, image_batch, original_image in [
            ("Left Eye",  lci, lp, lb, lo),
            ("Right Eye", rci, rp, rb, ro),
        ]:
            st.markdown(f"### 👁️ {eye_label} — Grad-CAM")
            try:
                heatmap = generate_gradcam_heatmap(image_batch, cls_idx, sel_layer)
                colored, overlay = apply_heatmap_to_image(
                    original_image, heatmap, colormap=sel_cmap, alpha=gc_alpha
                )

                cls = class_names[cls_idx]
                st.success(
                    f"Prediction: **{DISPLAY_NAMES[cls]}** ({probs[cls_idx]*100:.2f}%) "
                    f"· Layer: `{sel_layer}`"
                )

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.image(original_image, caption="Original", use_container_width=True)
                with c2:
                    st.image(colored, caption=f"Heatmap ({sel_cmap_label})", use_container_width=True)
                with c3:
                    st.image(overlay, caption="Overlay", use_container_width=True)

                st.download_button(
                    f"💾 Download {eye_label} Overlay",
                    data=pil_to_bytes(overlay),
                    file_name=f"gradcam_{eye_label.replace(' ','_').lower()}_{sel_layer}.png",
                    mime="image/png",
                    key=f"dl_{eye_label}",
                )

                foc = attention_focus(heatmap)
                st.info(
                    f"🎯 **Attention Focus Score: {foc['score']:.3f}** — "
                    f"{foc['label']}  |  {foc['desc']}"
                )

                with st.expander(f"📊 {eye_label} — Activation Distribution"):
                    fig_d = plot_activation_distribution(heatmap, eye_label)
                    st.pyplot(fig_d, use_container_width=True)
                    plt.close(fig_d)

            except Exception as exc:
                st.error(f"❌ Grad-CAM failed for {eye_label}: {exc}")

            st.markdown('<hr class="fdiv">', unsafe_allow_html=True)

        with st.expander("📚 All Compatible CNN Layers"):
            rows = []
            for i, ln in enumerate(GRADCAM_LAYERS, 1):
                lo_ = next((l for l in model.layers if l.name == ln), None)
                if lo_:
                    try:
                        shp = str(lo_.output.shape)
                        prm = f"{lo_.count_params():,}"
                    except:
                        shp = "N/A"
                        prm = "N/A"
                    rows.append({"#": i, "Name": ln, "Output Shape": shp, "Params": prm})
            st.dataframe(rows, hide_index=True, use_container_width=True)


# =============================================================================
# TAB — ANALYSIS
# =============================================================================

with tab_analysis:
    st.header("📊 Deep Image Analysis")

    if "left_prediction" not in st.session_state:
        st.info("Upload and classify both eyes in the **🔍 Predict** tab first.")
    else:
        (lci, lp, _, lo, _) = st.session_state["left_prediction"]
        (rci, rp, _, ro, _) = st.session_state["right_prediction"]
        lq = st.session_state["left_quality"]
        rq = st.session_state["right_quality"]

        st.subheader("🌈 RGB Channel Histograms")
        st.caption(
            "Pixel intensity distribution per colour channel. "
            "Reveals colour cast, overexposure, or unusual imaging conditions."
        )
        hc1, hc2 = st.columns(2)
        with hc1:
            fig = plot_rgb_histograms(lo, "Left Eye — RGB")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with hc2:
            fig = plot_rgb_histograms(ro, "Right Eye — RGB")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown('<hr class="fdiv">', unsafe_allow_html=True)

        st.subheader("📐 Quality Metrics — Side by Side")
        rows = [
            ("Resolution",     f"{lq['width']}×{lq['height']}",   f"{rq['width']}×{rq['height']}"),
            ("Total Pixels",   f"{lq['total_px']:,}",              f"{rq['total_px']:,}"),
            ("Brightness",     f"{lq['brightness']:.2f}",          f"{rq['brightness']:.2f}"),
            ("Contrast",       f"{lq['contrast']:.2f}",            f"{rq['contrast']:.2f}"),
            ("Sharpness",      f"{lq['sharpness']:.2f}",           f"{rq['sharpness']:.2f}"),
            ("SNR",            f"{lq['snr']:.2f}",                 f"{rq['snr']:.2f}"),
            ("Entropy (bits)", f"{lq['entropy']:.3f}",             f"{rq['entropy']:.3f}"),
            ("Overall Score",  f"{lq['overall_score']:.1f}/100",   f"{rq['overall_score']:.1f}/100"),
        ]
        table_rows = "".join(
            f"<tr><td><strong>{m}</strong></td><td>{l}</td><td>{r}</td></tr>"
            for m, l, r in rows
        )
        st.markdown(
            f'<table class="ctable"><thead><tr>'
            f'<th>Metric</th><th>👁️ Left Eye</th><th>👁️ Right Eye</th>'
            f'</tr></thead><tbody>{table_rows}</tbody></table>',
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="fdiv">', unsafe_allow_html=True)

        st.subheader("🔄 Prediction Stability (Counterfactual Analysis)")
        sc1, sc2 = st.columns(2)
        for col, cls_idx, probs, eye_lbl in [
            (sc1, lci, lp, "Left Eye"),
            (sc2, rci, rp, "Right Eye"),
        ]:
            with col:
                cf = counterfactual(probs, cls_idx)
                st.markdown(f"**{eye_lbl}**")
                st.markdown(f"**{cf['stability']}** — margin `{cf['margin']:.1f}%`")
                st.markdown(f"Runner-up: **{cf['second_name']}** at `{cf['second_conf']:.1f}%`")
                st.caption(cf["desc"])


# =============================================================================
# TAB — BILATERAL
# =============================================================================

with tab_bilateral:
    st.header("⚖️ Bilateral Eye Comparison")
    st.markdown(
        "Side-by-side comparison of model outputs for both eyes, "
        "including inter-eye symmetry scoring."
    )

    if "left_prediction" not in st.session_state:
        st.info("Upload both eyes in the **🔍 Predict** tab first.")
    else:
        (lci, lp, _, lo, _) = st.session_state["left_prediction"]
        (rci, rp, _, ro, _) = st.session_state["right_prediction"]
        lc    = class_names[lci]
        rc    = class_names[rci]
        lconf = lp[lci] * 100
        rconf = rp[rci] * 100

        bl, br = st.columns(2)
        for col, eye_lbl, cls, conf, img in [
            (bl, "Left Eye",  lc, lconf, lo),
            (br, "Right Eye", rc, rconf, ro),
        ]:
            with col:
                st.image(img, caption=eye_lbl, use_container_width=True)
                is_dis = cls != "normal"
                card_c = "rcard-disease" if is_dis else "rcard-normal"
                bdg_c  = {"low": "badge-low", "medium": "badge-med", "high": "badge-high"}[CLASS_RISK[cls][0]]
                st.markdown(f"""
                <div class="rcard {card_c}">
                  <div style="font-size:1.2rem;font-weight:700;color:#0f172a;">
                    {CLASS_ICONS[cls]} {DISPLAY_NAMES[cls]}
                  </div>
                  <div style="color:#475569;margin:4px 0;">
                    Confidence: <strong>{conf:.2f}%</strong>
                  </div>
                  <span class="badge {bdg_c}">{CLASS_RISK[cls][0].upper()} RISK</span>
                </div>
                """, unsafe_allow_html=True)
                st.metric("Confidence", f"{conf:.2f}%")

        st.markdown('<hr class="fdiv">', unsafe_allow_html=True)

        sym = bilateral_symmetry(lp, rp, lci, rci)
        bdg_c = {"low": "badge-low", "medium": "badge-med", "high": "badge-high"}[sym["risk"]]
        st.subheader("🔀 Bilateral Symmetry Analysis")
        st.markdown(
            f'<span class="badge {bdg_c}">{sym["label"]}</span>'
            f'<div style="margin-top:8px;color:#374151;">{sym["desc"]}</div>',
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Symmetry Score", f"{sym['sym_score']:.1f}%")
        with m2: st.metric("KL Divergence",  f"{sym['kl']:.4f}")
        with m3: st.metric("Agreement",      "✅ Yes" if sym["agree"] else "❌ No")

        st.markdown("**Per-class probability difference (Left − Right):**")
        fig_d, ax_d = plt.subplots(figsize=(6, 2.4))
        diff_vals = (lp - rp) * 100
        colors_d  = ["#22c55e" if v >= 0 else "#ef4444" for v in diff_vals]
        ax_d.bar([DISPLAY_NAMES[c] for c in class_names], diff_vals,
                 color=colors_d, edgecolor="none", alpha=0.85)
        ax_d.axhline(0, color="#94a3b8", linewidth=1)
        ax_d.set_ylabel("L − R (%)", fontsize=8, color="#475569")
        ax_d.set_title("Left − Right Probability Difference", fontsize=9, fontweight="600")
        ax_d.tick_params(labelsize=8)
        ax_d.spines["top"].set_visible(False)
        ax_d.spines["right"].set_visible(False)
        fig_d.patch.set_facecolor("white")
        ax_d.set_facecolor("white")
        plt.tight_layout(pad=0.4)
        st.pyplot(fig_d, use_container_width=True)
        plt.close(fig_d)

        st.markdown('<hr class="fdiv">', unsafe_allow_html=True)
        st.subheader("🩺 Clinical Guidance")
        if lc == rc:
            if lc == "normal":
                st.success(f"✅ Both eyes: **Normal**. {CLASS_RISK['normal'][1]}.")
            else:
                st.warning(f"⚠️ Both eyes: **{DISPLAY_NAMES[lc]}**. {CLASS_RISK[lc][1]}.")
        else:
            st.warning(
                f"⚠️ **Asymmetric findings** — Left: {DISPLAY_NAMES[lc]}, "
                f"Right: {DISPLAY_NAMES[rc]}. Clinical review recommended."
            )
        st.caption("AI output only. Always consult a qualified ophthalmologist.")


# =============================================================================
# TAB — REPORT
# =============================================================================

with tab_report:
    st.header("📋 Diagnostic Report")

    if "left_prediction" not in st.session_state:
        st.info("Upload both eyes in the **🔍 Predict** tab first.")
    else:
        (lci, lp, _, _, _) = st.session_state["left_prediction"]
        (rci, rp, _, _, _) = st.session_state["right_prediction"]
        lq = st.session_state.get("left_quality")
        rq = st.session_state.get("right_quality")

        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Left Eye",        DISPLAY_NAMES[class_names[lci]])
        with m2: st.metric("Left Confidence", f"{lp[lci]*100:.2f}%")
        with m3: st.metric("Right Eye",        DISPLAY_NAMES[class_names[rci]])
        with m4: st.metric("Right Confidence", f"{rp[rci]*100:.2f}%")

        st.markdown('<hr class="fdiv">', unsafe_allow_html=True)
        st.subheader("📊 Detailed Probabilities")
        pc1, pc2 = st.columns(2)
        for col, probs, cls_idx, eye_lbl in [
            (pc1, lp, lci, "Left Eye"),
            (pc2, rp, rci, "Right Eye"),
        ]:
            with col:
                st.markdown(f"**{eye_lbl}**")
                for i, p in enumerate(probs):
                    marker = " ⭐" if i == cls_idx else ""
                    st.progress(float(p),
                                text=f"{DISPLAY_NAMES[class_names[i]]}: {p*100:.2f}%{marker}")

        st.markdown('<hr class="fdiv">', unsafe_allow_html=True)
        st.subheader("📥 Download Report")

        d1, d2 = st.columns(2)
        with d1:
            txt = text_report(lci, lp, rci, rp)
            with st.expander("👀 Preview Text Report"):
                st.text(txt)
            st.download_button(
                "📄 Download Text Report (.txt)", data=txt,
                file_name="ocular_report.txt", mime="text/plain",
            )
        with d2:
            jsn = json_report(lci, lp, rci, rp, lq, rq)
            with st.expander("👀 Preview JSON Report"):
                st.code(jsn, language="json")
            st.download_button(
                "🗂️ Download JSON Report (.json)", data=jsn,
                file_name="ocular_report.json", mime="application/json",
            )


# =============================================================================
# TAB — FOLLOW-UP SCHEDULING  (NEW)
# =============================================================================

with tab_followup:
    st.header("📅 Follow-Up Scheduling & Clinical Recommendations")
    st.markdown("""
    Structured follow-up recommendations based on the model's prediction, confidence level,
    and image quality — derived from published clinical screening guidelines.

    **Sources:** American Academy of Ophthalmology (AAO) Preferred Practice Patterns ·
    American Diabetes Association (ADA) Standards of Care 2023 · WHO Global Eye Health Guidelines.
    """)
    st.error(
        "⚠️ These recommendations are AI-generated for **educational purposes only**. "
        "They are NOT a substitute for clinical judgement. "
        "Always follow the advice of a qualified ophthalmologist or physician."
    )

    if "left_prediction" not in st.session_state:
        st.info("👈 Upload and classify both eyes in the **🔍 Predict** tab first.")
    else:
        (lci, lp, _, _, _) = st.session_state["left_prediction"]
        (rci, rp, _, _, _) = st.session_state["right_prediction"]
        lq = st.session_state.get("left_quality",  {"overall_score": 75})
        rq = st.session_state.get("right_quality", {"overall_score": 75})
        lc = class_names[lci]
        rc = class_names[rci]

        # Summary metrics
        sm1, sm2, sm3, sm4 = st.columns(4)
        with sm1: st.metric("Left Eye",        DISPLAY_NAMES[lc])
        with sm2: st.metric("Left Confidence", f"{lp[lci]*100:.1f}%")
        with sm3: st.metric("Right Eye",        DISPLAY_NAMES[rc])
        with sm4: st.metric("Right Confidence", f"{rp[rci]*100:.1f}%")

        st.markdown('<hr class="fdiv">', unsafe_allow_html=True)

        # Left eye card
        fu_left = generate_followup_recommendation(lc, lp[lci]*100, lq["overall_score"])
        display_followup_card(fu_left, "Left Eye")

        st.markdown('<hr class="fdiv">', unsafe_allow_html=True)

        # Right eye card
        fu_right = generate_followup_recommendation(rc, rp[rci]*100, rq["overall_score"])
        display_followup_card(fu_right, "Right Eye")

        st.markdown('<hr class="fdiv">', unsafe_allow_html=True)

        # Overall priority triage
        st.subheader("🚦 Overall Priority Triage")
        overall_risk = (
            "high"   if fu_left["tier"] == "high"   or fu_right["tier"] == "high"
            else "medium" if fu_left["tier"] == "medium" or fu_right["tier"] == "medium"
            else "low"
        )
        risk_messages = {
            "high": (
                "🔴 HIGH PRIORITY", "#dc2626",
                "At least one eye carries high-risk findings. "
                "Seek ophthalmological evaluation promptly. Early intervention is critical.",
            ),
            "medium": (
                "🟡 MODERATE PRIORITY", "#a16207",
                "Moderate-risk findings detected. "
                "Schedule a specialist appointment within the recommended timeframe.",
            ),
            "low": (
                "🟢 ROUTINE", "#15803d",
                "No high-risk findings detected. "
                "Continue routine annual eye examinations.",
            ),
        }
        msg_label, msg_color, msg_text = risk_messages[overall_risk]
        st.markdown(
            f'<div style="background:{msg_color}11;border:2px solid {msg_color}44;'
            f'border-radius:12px;padding:1.2rem 1.4rem;">'
            f'<div style="font-size:1.1rem;font-weight:700;color:{msg_color};">{msg_label}</div>'
            f'<div style="color:#374151;font-size:.9rem;margin-top:6px;">{msg_text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="fdiv">', unsafe_allow_html=True)
        st.caption(
            "Guideline sources: AAO Preferred Practice Patterns (2019–2021), "
            "ADA Standards of Medical Care in Diabetes 2023, "
            "WHO World Report on Vision 2019."
        )


# =============================================================================
# TAB — VISUALIZATIONS
# =============================================================================

with tab_viz:
    st.header("📈 Model Training Visualizations")
    st.markdown("Charts generated during model development and evaluation.")

    VIS = [
        ("01_distribution.png",   "1. Dataset Class Distribution",
         "Distribution of images across classes in train/val/test sets."),
        ("02_train_test.png",      "2. Train / Val / Test Split",
         "Pie charts showing dataset partition ratios."),
        ("03_model_building.png",  "3. Sample Training Images",
         "Representative retinal images used during training."),
        ("04_evaluate.png",        "4. Training & Validation Curves",
         "Accuracy and loss curves across epochs."),
        ("05_confusion_matrix.png","5. Confusion Matrix",
         "Per-class classification performance on the test set."),
        ("06_roc_curve.png",       "6. ROC-AUC Curves",
         "Multi-class ROC curves measuring discrimination ability."),
        ("07_comparison.png",      "7. Precision · Recall · F1-Score",
         "Per-class metrics comparison."),
    ]
    for fname, title, desc in VIS:
        st.subheader(title)
        st.caption(desc)
        fpath = os.path.join(VISUALIZATION_DIR, fname)
        if os.path.isfile(fpath):
            st.image(fpath, use_container_width=True)
        else:
            st.warning(f"File not found: `{fname}`")
        st.markdown('<hr class="fdiv">', unsafe_allow_html=True)


# =============================================================================
# TAB — DISEASE LIBRARY
# =============================================================================

with tab_disease:
    st.header("📚 Ocular Disease Library")
    st.markdown(
        "Comprehensive educational reference for each condition classified by this model. "
        "**Not a substitute for professional medical advice.**"
    )

    bc = st.columns(4)
    for col, key in zip(bc, class_names):
        with col:
            if st.button(
                f"{CLASS_ICONS[key]} {DISPLAY_NAMES[key]}",
                key=f"dbtn_{key}", use_container_width=True
            ):
                st.session_state["sel_disease"] = key

    ds   = st.session_state["sel_disease"]
    info = DISEASE_INFO[ds]
    clr  = CLASS_COLORS_MPL[ds]

    st.markdown('<hr class="fdiv">', unsafe_allow_html=True)
    st.markdown(
        f'<div style="padding:14px 18px;background:#fff;border-radius:12px;'
        f'border-left:5px solid {clr};margin-bottom:1rem;">'
        f'<div style="font-size:1.4rem;font-weight:700;color:#0f172a;">{info["title"]}</div>'
        f'<div style="color:#374151;font-size:.9rem;line-height:1.65;margin-top:6px;">'
        f'{info["description"]}</div></div>',
        unsafe_allow_html=True,
    )

    q1, q2, q3, q4 = st.columns(4)
    with q1: st.metric("Global Prevalence",   info["prevalence"])
    with q2: st.metric("Screening Frequency", info["screening_freq"])
    with q3: st.metric("ICD-10 Code",         info["icd_code"])
    with q4: st.metric("AI Risk Level",       CLASS_RISK[ds][0].upper())

    st.markdown('<hr class="fdiv">', unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("### ⚠️ Common Symptoms")
        for s in info["symptoms"]:
            st.markdown(
                f'<div class="titem"><span class="ttext">• {s}</span></div>',
                unsafe_allow_html=True,
            )
    with sc2:
        st.markdown("### 📌 Risk Factors")
        for r in info["risk_factors"]:
            st.markdown(
                f'<div class="titem"><span class="ttext">• {r}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="fdiv">', unsafe_allow_html=True)
    st.markdown("### 🩺 Management & Treatment")
    st.info(info["management"])
    st.warning(
        "**When to seek urgent care?** Sudden vision loss, new floaters/flashes, "
        "severe eye pain, or rapidly changing vision warrant immediate attention. "
        "Do not use this AI tool to delay professional evaluation."
    )


# =============================================================================
# TAB — HOW IT WORKS
# =============================================================================

with tab_how:
    st.header("⚙️ Technical Pipeline")

    steps = [
        ("🖼️ Image Upload",
         "User uploads JPG/PNG retinal fundus photographs for both eyes."),
        ("🔍 Quality Assessment",
         "Brightness, contrast, sharpness, SNR, and entropy are measured. "
         "Poor quality images reduce effective confidence in risk stratification."),
        ("🧪 Preprocessing Preview",
         "Four views are shown: original (224×224), contrast-enhanced, grayscale, "
         "and edge-detected — giving insight into what the model processes."),
        ("📐 Normalisation",
         "Images are resized to 224×224 and pixel values normalised from [0–255] to [0.0–1.0]."),
        ("🧠 CNN Inference",
         "The custom CNN processes the image through conv, pooling, flatten, dense, "
         "and softmax layers, producing a 4-class probability vector."),
        ("📊 Probability Output",
         "Softmax scores for Glaucoma, Normal, Cataract, and Diabetic Retinopathy. "
         "Highest probability class is selected."),
        ("🔬 Grad-CAM",
         "A separate differentiable model graph is constructed for the selected conv layer. "
         "Gradient-weighted feature maps are computed, ReLU'd, normalised, and overlaid."),
        ("⚖️ Bilateral Analysis",
         "Left and right eye predictions are compared using symmetry score, "
         "KL-divergence, and per-class difference charts."),
        ("🚦 Risk Stratification",
         "Each prediction is mapped to Low / Medium / High risk with a follow-up action."),
        ("📅 Follow-Up Scheduling",
         "Effective confidence (adjusted for image quality) determines a tier (high/medium/low). "
         "AAO/ADA guideline-based referral, screening dates, lifestyle, and monitoring "
         "recommendations are generated per eye."),
        ("📋 Report Generation",
         "Structured .txt and .json reports are produced, including all probabilities, "
         "quality scores, bilateral findings, and recommended actions."),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(
            f'<div class="titem">'
            f'<span class="tlabel">Step {i:02d}</span>'
            f'<div><strong>{title}</strong><br>'
            f'<span class="ttext">{desc}</span></div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="fdiv">', unsafe_allow_html=True)
    st.subheader("🧠 Model Architecture")
    st.markdown("""
- Multiple **Conv2D** blocks with ReLU activations
- **MaxPooling2D** layers for spatial downsampling
- **Flatten** → **Dense** (ReLU) → **Dropout**
- **Dense (Softmax)** — 4 output neurons, one per class
""")
    if GRADCAM_LAYERS:
        st.markdown(f"**Grad-CAM compatible layers:** `{', '.join(GRADCAM_LAYERS)}`")

    st.markdown('<hr class="fdiv">', unsafe_allow_html=True)
    st.subheader("📦 Tech Stack")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("**Model & Inference**\n- TensorFlow / Keras\n- Custom CNN\n- Hugging Face Hub")
    with t2:
        st.markdown("**Explainability**\n- Grad-CAM (custom graph)\n- Multi-layer support\n- 6 colormaps")
    with t3:
        st.markdown("**App & Visualisation**\n- Streamlit\n- Matplotlib\n- Pillow / SciPy")


# =============================================================================
# FOOTER
# =============================================================================

st.markdown('<hr class="fdiv">', unsafe_allow_html=True)
st.markdown("""
<div style="background:#fff5f5;border:1px solid #fca5a5;border-radius:12px;
            padding:1.2rem 1.6rem;margin-top:1rem;">
  <strong style="color:#dc2626;">⚠️ Medical Disclaimer</strong>
  <p style="color:#374151;font-size:.87rem;line-height:1.7;margin:.5rem 0 0;">
    This application is for <strong>educational and demonstration purposes only</strong>.
    It is <strong>NOT</strong> a substitute for professional medical advice, diagnosis, or treatment.
    Do not use it for self-diagnosis. Always consult a qualified ophthalmologist or optometrist.
    Follow-up recommendations are AI-generated from published guidelines and do not replace
    clinical judgement. Grad-CAM heatmaps show model attention, not anatomical disease locations.
    Image quality, dataset limitations, and rare conditions all affect model accuracy.
  </p>
</div>
""", unsafe_allow_html=True)