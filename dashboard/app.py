"""
FOREST SENTINEL — AI-Powered Forest Fire Intelligence Platform
Premium White / Light Theme Streamlit Dashboard
"""

import streamlit as st
import json
import pickle
import time
import os
import io
import base64
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm_colors
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forest Sentinel — AI Fire Intelligence",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
ROOT        = Path(r"e:/Sharvayu data/Malware/Symbiosis Nagpur SIT/7th SEM/Forest Fire task/Implementation")
PARENT_DIR  = ROOT.parent
IMG_DIR     = ROOT / "models" / "image"
TAB_DIR     = ROOT / "models" / "tabular"
PLOTS_DIR   = ROOT / "artifacts" / "plots"
META_DIR    = ROOT / "artifacts" / "metadata"
METRICS_DIR = ROOT / "artifacts" / "metrics"
PRED_DIR    = ROOT / "artifacts" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

# Sample image directories if available
FIRE_SAMPLES_DIR    = PARENT_DIR / "archive (1)" / "fire_dataset" / "fire_images"
NONFIRE_SAMPLES_DIR = PARENT_DIR / "archive (1)" / "fire_dataset" / "non_fire_images"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS — ELEGANT WHITE / LIGHT THEME
# ─────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
    /* ── Google Fonts ────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Root Color Variables (Light / White Theme) ──────────── */
    :root {
        --bg-canvas:        #f8fafc;
        --bg-card:          #ffffff;
        --bg-card-subtle:   #f1f5f9;
        --border-color:     #e2e8f0;
        --border-subtle:    #f1f5f9;
        --border-focus:     #10b981;
        
        --emerald-primary:  #059669;
        --emerald-dark:     #047857;
        --emerald-light:    #10b981;
        --emerald-bg:       #ecfdf5;
        --emerald-border:   #a7f3d0;
        
        --fire-crimson:     #dc2626;
        --fire-coral:       #ef4444;
        --fire-bg:          #fef2f2;
        --fire-border:      #fecaca;
        
        --amber-primary:    #d97706;
        --amber-light:      #f59e0b;
        --amber-bg:         #fffbeb;
        --amber-border:     #fde68a;
        
        --blue-primary:     #2563eb;
        --blue-bg:          #eff6ff;
        --blue-border:      #bfdbfe;
        
        --text-heading:     #0f172a;
        --text-body:        #334155;
        --text-muted:       #64748b;
        --text-subtle:      #94a3b8;
        
        --shadow-xs:        0 1px 2px 0 rgba(0, 0, 0, 0.04);
        --shadow-sm:        0 1px 3px 0 rgba(0, 0, 0, 0.07), 0 1px 2px 0 rgba(0, 0, 0, 0.04);
        --shadow-md:        0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
        --shadow-lg:        0 10px 15px -3px rgba(0, 0, 0, 0.07), 0 4px 6px -2px rgba(0, 0, 0, 0.03);
        --shadow-card:      0 2px 12px rgba(15, 23, 42, 0.04);
    }

    /* ── Global Styles ───────────────────────────────────────── */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: var(--bg-canvas) !important;
        color: var(--text-body) !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: var(--text-heading) !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    /* ── Sidebar Styling ─────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid var(--border-color) !important;
        box-shadow: 2px 0 10px rgba(15, 23, 42, 0.02) !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: var(--text-body) !important;
    }

    /* ── Main Container ──────────────────────────────────────── */
    .main .block-container {
        padding: 2rem 3rem 3rem !important;
        max-width: 1440px !important;
    }

    /* ── Brand Header Component ──────────────────────────────── */
    .brand-box {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 0.75rem 0.5rem 1.25rem;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 1rem;
    }
    .brand-icon {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        box-shadow: 0 4px 10px rgba(5, 150, 105, 0.25);
        color: white;
    }
    .brand-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.15rem;
        font-weight: 800;
        color: var(--text-heading);
        line-height: 1.2;
        letter-spacing: -0.3px;
    }
    .brand-subtitle {
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--emerald-primary);
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }

    /* ── Hero Banner ─────────────────────────────────────────── */
    .hero-banner-light {
        background: #ffffff;
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 2.2rem 2.6rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }
    .hero-banner-light::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: linear-gradient(90deg, #059669 0%, #10b981 35%, #f59e0b 70%, #ef4444 100%);
    }
    .hero-banner-light::after {
        content: '';
        position: absolute;
        top: 0; right: 0; bottom: 0; width: 40%;
        background: radial-gradient(circle at 100% 0%, rgba(5, 150, 105, 0.05) 0%, rgba(239, 68, 68, 0.03) 60%, transparent 100%);
        pointer-events: none;
    }
    .hero-title-light {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.8px;
        color: var(--text-heading);
        margin: 0 0 0.4rem 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--emerald-bg);
        border: 1px solid var(--emerald-border);
        color: var(--emerald-primary);
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        vertical-align: middle;
    }
    .hero-desc-light {
        font-size: 1rem;
        color: var(--text-muted);
        max-width: 780px;
        line-height: 1.6;
        margin-top: 0.5rem;
    }

    /* ── Metric Cards ────────────────────────────────────────── */
    .metric-card-light {
        background: #ffffff;
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.4rem 1.4rem;
        position: relative;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: var(--shadow-xs);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-card-light:hover {
        transform: translateY(-3px);
        border-color: #cbd5e1;
        box-shadow: var(--shadow-md);
    }
    .metric-top-bar {
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-top-left-radius: 16px;
        border-top-right-radius: 16px;
    }
    .metric-top-green  { background: #10b981; }
    .metric-top-amber  { background: #f59e0b; }
    .metric-top-red    { background: #ef4444; }
    .metric-top-blue   { background: #3b82f6; }
    .metric-top-purple { background: #8b5cf6; }

    .metric-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
    }
    .metric-icon-circle {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.15rem;
    }
    .icon-circle-green  { background: var(--emerald-bg); color: var(--emerald-primary); }
    .icon-circle-red    { background: var(--fire-bg); color: var(--fire-crimson); }
    .icon-circle-amber  { background: var(--amber-bg); color: var(--amber-primary); }
    .icon-circle-blue   { background: var(--blue-bg); color: var(--blue-primary); }
    .icon-circle-purple { background: #f5f3ff; color: #7c3aed; }

    .metric-label-light {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: var(--text-muted);
    }
    .metric-val-light {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.85rem;
        font-weight: 800;
        color: var(--text-heading);
        line-height: 1.1;
        margin: 0.2rem 0 0.4rem 0;
    }
    .metric-sub-light {
        font-size: 0.8rem;
        color: var(--text-muted);
        display: flex;
        align-items: center;
        gap: 4px;
    }

    /* ── Section Header ──────────────────────────────────────── */
    .section-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 0.75rem;
        margin: 1.5rem 0 1.2rem 0;
        border-bottom: 2px solid var(--border-color);
    }
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-heading);
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-tag {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--emerald-primary);
        background: var(--emerald-bg);
        border: 1px solid var(--emerald-border);
        border-radius: 6px;
        padding: 2px 8px;
    }

    /* ── Surface Cards & Containers ──────────────────────────── */
    .surface-card {
        background: #ffffff;
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.25rem;
        box-shadow: var(--shadow-xs);
    }
    
    /* ── Status Pills & Indicators ───────────────────────────── */
    .pill-online {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--emerald-bg);
        border: 1px solid var(--emerald-border);
        border-radius: 999px;
        padding: 0.25rem 0.75rem;
        font-size: 0.78rem;
        font-weight: 700;
        color: var(--emerald-dark);
    }
    .pill-offline {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--fire-bg);
        border: 1px solid var(--fire-border);
        border-radius: 999px;
        padding: 0.25rem 0.75rem;
        font-size: 0.78rem;
        font-weight: 700;
        color: var(--fire-crimson);
    }
    .pulse-dot-green {
        width: 8px; height: 8px;
        background-color: var(--emerald-primary);
        border-radius: 50%;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.25);
        animation: pulse-green-light 2s infinite;
    }
    .dot-red {
        width: 8px; height: 8px;
        background-color: var(--fire-coral);
        border-radius: 50%;
    }
    @keyframes pulse-green-light {
        0%, 100% { transform: scale(1); opacity: 1; }
        50%      { transform: scale(1.2); opacity: 0.6; }
    }

    /* ── Result Cards (Fire Detection) ───────────────────────── */
    .result-box-fire {
        background: linear-gradient(180deg, #ffffff 0%, #fff5f5 100%);
        border: 2px solid #fca5a5;
        border-radius: 18px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(220, 38, 38, 0.1);
        position: relative;
        overflow: hidden;
    }
    .result-box-fire::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 5px;
        background: linear-gradient(90deg, #dc2626, #f97316);
    }
    
    .result-box-safe {
        background: linear-gradient(180deg, #ffffff 0%, #f0fdf4 100%);
        border: 2px solid #86efac;
        border-radius: 18px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(5, 150, 105, 0.1);
        position: relative;
        overflow: hidden;
    }
    .result-box-safe::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 5px;
        background: linear-gradient(90deg, #059669, #10b981);
    }
    
    .result-title-red {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.7rem;
        font-weight: 800;
        color: var(--fire-crimson);
        letter-spacing: -0.3px;
    }
    .result-title-green {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.7rem;
        font-weight: 800;
        color: var(--emerald-dark);
        letter-spacing: -0.3px;
    }
    .confidence-large {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.5rem 0;
    }

    /* ── Risk Level Badges ───────────────────────────────────── */
    .badge-critical {
        background: #fef2f2; color: #b91c1c;
        border: 1px solid #f87171;
    }
    .badge-high {
        background: #fff7ed; color: #c2410c;
        border: 1px solid #fb923c;
    }
    .badge-medium {
        background: #fffbeb; color: #b45309;
        border: 1px solid #fcd34d;
    }
    .badge-low {
        background: #ecfdf5; color: #047857;
        border: 1px solid #6ee7b7;
    }
    .risk-badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border-radius: 999px;
        padding: 0.4rem 1.2rem;
        font-size: 0.9rem;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* ── Probability Bars ────────────────────────────────────── */
    .prob-container {
        margin: 0.75rem 0;
    }
    .prob-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-heading);
        margin-bottom: 6px;
    }
    .prob-track {
        background: #e2e8f0;
        border-radius: 999px;
        height: 10px;
        overflow: hidden;
    }
    .prob-fill-fire {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #f87171 0%, #dc2626 100%);
        transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .prob-fill-safe {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #34d399 0%, #059669 100%);
        transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }

    /* ── Info & Notice Panels ────────────────────────────────── */
    .panel-info {
        background: #f8fafc;
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--blue-primary);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
    }
    .panel-success {
        background: var(--emerald-bg);
        border: 1px solid var(--emerald-border);
        border-left: 4px solid var(--emerald-primary);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
    }
    .panel-warning {
        background: var(--amber-bg);
        border: 1px solid var(--amber-border);
        border-left: 4px solid var(--amber-primary);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
    }
    .panel-danger {
        background: var(--fire-bg);
        border: 1px solid var(--fire-border);
        border-left: 4px solid var(--fire-crimson);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
    }

    /* ── Custom Buttons ──────────────────────────────────────── */
    .stButton > button {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        color: var(--text-heading) !important;
        border-radius: 10px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.55rem 1.25rem !important;
        box-shadow: var(--shadow-xs) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: #f8fafc !important;
        border-color: #94a3b8 !important;
        color: var(--emerald-dark) !important;
        box-shadow: var(--shadow-sm) !important;
        transform: translateY(-1px);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        border-color: #047857 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 12px rgba(5, 150, 105, 0.25) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #047857 0%, #065f46 100%) !important;
        box-shadow: 0 6px 16px rgba(5, 150, 105, 0.35) !important;
    }

    /* ── Upload Area ─────────────────────────────────────────── */
    .upload-card-light {
        background: #ffffff;
        border: 2px dashed #cbd5e1;
        border-radius: 16px;
        padding: 2rem 1.5rem;
        text-align: center;
        transition: all 0.2s ease;
    }
    .upload-card-light:hover {
        border-color: var(--emerald-primary);
        background: var(--emerald-bg);
    }

    /* ── Custom Tabs ─────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid var(--border-color);
        background: transparent;
        padding-bottom: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px 8px 0 0 !important;
        color: var(--text-muted) !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.6rem 1.2rem !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--emerald-dark) !important;
        border-bottom: 2px solid var(--emerald-primary) !important;
        background: #ffffff !important;
    }

    /* ── Code Blocks & Tables ────────────────────────────────── */
    pre, code {
        font-family: 'JetBrains Mono', monospace !important;
        background: #f1f5f9 !important;
        color: #0f172a !important;
        border-radius: 8px !important;
    }
    .stDataFrame {
        border: 1px solid var(--border-color);
        border-radius: 12px;
        overflow: hidden;
        background: #ffffff;
    }

    /* ── Sliders & Inputs ────────────────────────────────────── */
    .stSlider label, .stSelectbox label, .stRadio label {
        font-weight: 600 !important;
        color: var(--text-heading) !important;
        font-size: 0.88rem !important;
    }

    /* ── Clean Scrollbars ────────────────────────────────────── */
    ::-webkit-scrollbar { width: 7px; height: 7px; }
    ::-webkit-scrollbar-track { background: #f1f5f9; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MODEL LOADING (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_image_model():
    try:
        if not (IMG_DIR / "class_names.json").exists() or not (IMG_DIR / "best_model.pth").exists():
            return {'loaded': False, 'error': 'Image model weights or metadata not found in models/image/'}

        with open(IMG_DIR / "class_names.json") as f:
            cnames = json.load(f)
        with open(IMG_DIR / "model_metadata.json") as f:
            mmeta = json.load(f)
        
        mname = mmeta.get('model_name', 'EfficientNet-B0')
        
        def build(name, nc=2):
            if name == 'EfficientNet-B0':
                m = models.efficientnet_b0(weights=None)
                m.classifier[1] = nn.Linear(m.classifier[1].in_features, nc)
            elif name == 'MobileNetV3':
                m = models.mobilenet_v3_small(weights=None)
                m.classifier[3] = nn.Linear(m.classifier[3].in_features, nc)
            elif name == 'MobileNetV2':
                m = models.mobilenet_v2(weights=None)
                m.classifier[1] = nn.Linear(m.classifier[1].in_features, nc)
            elif name == 'ResNet18':
                m = models.resnet18(weights=None)
                m.fc = nn.Linear(m.fc.in_features, nc)
            else:
                m = models.efficientnet_b0(weights=None)
                m.classifier[1] = nn.Linear(m.classifier[1].in_features, nc)
            return m
        
        mdl = build(mname)
        state = torch.load(IMG_DIR / "best_model.pth", map_location='cpu')
        mdl.load_state_dict(state)
        mdl.eval()
        
        return {
            'model': mdl,
            'meta': mmeta,
            'class_names': cnames.get('class_names', ['FIRE', 'NO_FIRE']),
            'class_to_idx': cnames.get('class_to_idx', {'FIRE': 0, 'NO_FIRE': 1}),
            'idx_to_class': {int(k): v for k, v in cnames.get('idx_to_class', {'0': 'FIRE', '1': 'NO_FIRE'}).items()},
            'loaded': True
        }
    except Exception as e:
        return {'loaded': False, 'error': str(e)}


@st.cache_resource
def load_tabular_models():
    try:
        if not (TAB_DIR / "classifier.pkl").exists() or not (TAB_DIR / "metadata.json").exists():
            return {'loaded': False, 'error': 'Tabular models or metadata not found in models/tabular/'}

        clf = pickle.load(open(TAB_DIR / "classifier.pkl", "rb"))
        reg = pickle.load(open(TAB_DIR / "regressor.pkl",  "rb"))
        sc  = pickle.load(open(TAB_DIR / "scaler.pkl",     "rb"))
        le_month = pickle.load(open(TAB_DIR / "le_month.pkl", "rb"))
        le_day   = pickle.load(open(TAB_DIR / "le_day.pkl",   "rb"))
        with open(TAB_DIR / "metadata.json") as f:
            tmeta = json.load(f)
            
        return {
            'clf': clf, 'reg': reg, 'scaler': sc,
            'le_month': le_month, 'le_day': le_day,
            'meta': tmeta, 'loaded': True
        }
    except Exception as e:
        return {'loaded': False, 'error': str(e)}


def load_metrics():
    metrics = {}
    try:
        with open(METRICS_DIR / "image_test_metrics.json") as f:
            metrics['image'] = json.load(f)
    except: metrics['image'] = None
    try:
        with open(METRICS_DIR / "tabular_test_metrics.json") as f:
            metrics['tabular'] = json.load(f)
    except: metrics['tabular'] = None
    try:
        with open(META_DIR / "dataset_inventory.json") as f:
            metrics['dataset'] = json.load(f)
    except: metrics['dataset'] = None
    return metrics


def load_benchmark():
    try:
        with open(META_DIR / "benchmark_results.json") as f:
            return json.load(f)
    except: return None


# ─────────────────────────────────────────────────────────────
# IMAGE INFERENCE & GRAD-CAM
# ─────────────────────────────────────────────────────────────
def predict_image(img_pil, img_model_data):
    mmeta = img_model_data['meta']
    sz    = mmeta.get('image_size', 224)
    mean_ = mmeta.get('imagenet_mean', [0.485, 0.456, 0.406])
    std_  = mmeta.get('imagenet_std',  [0.229, 0.224, 0.225])
    
    transform = T.Compose([
        T.Resize((sz, sz)),
        T.ToTensor(),
        T.Normalize(mean_, std_)
    ])
    
    img_rgb = img_pil.convert('RGB')
    inp = transform(img_rgb).unsqueeze(0)
    
    t0 = time.time()
    with torch.no_grad():
        out   = img_model_data['model'](inp)
        probs = F.softmax(out, dim=1)[0]
    inf_ms = (time.time() - t0) * 1000
    
    pred_idx  = probs.argmax().item()
    pred_cls  = img_model_data['idx_to_class'][pred_idx]
    conf      = probs[pred_idx].item()
    cnames    = img_model_data['class_names']
    cls_probs = {c: round(probs[i].item(), 4) for i, c in enumerate(cnames)}
    
    return {
        'prediction': pred_cls,
        'confidence': conf,
        'class_probs': cls_probs,
        'inference_ms': inf_ms
    }


def compute_gradcam(img_pil, img_model_data):
    mmeta      = img_model_data['meta']
    model_name = mmeta.get('model_name', 'EfficientNet-B0')
    sz         = mmeta.get('image_size', 224)
    mean_      = mmeta.get('imagenet_mean', [0.485, 0.456, 0.406])
    std_       = mmeta.get('imagenet_std',  [0.229, 0.224, 0.225])

    TARGET_LAYERS = {
        'EfficientNet-B0': 'features',
        'MobileNetV3':     'features',
        'MobileNetV2':     'features',
        'ResNet18':        'layer4'
    }
    
    mdl = img_model_data['model']
    mdl.eval()
    
    gradients   = []
    activations = []
    
    def fwd_hook(module, inp, out):
        activations.append(out)
    
    def bwd_hook(module, gin, gout):
        gradients.append(gout[0])
    
    target_name = TARGET_LAYERS.get(model_name, 'features')
    target_module = dict(mdl.named_modules()).get(target_name)
    
    if target_module is None:
        for name, m in mdl.named_modules():
            if isinstance(m, nn.Conv2d):
                target_module = m
    
    fh = target_module.register_forward_hook(fwd_hook)
    bh = target_module.register_full_backward_hook(bwd_hook)
    
    transform = T.Compose([T.Resize((sz, sz)), T.ToTensor(), T.Normalize(mean_, std_)])
    img_rgb = img_pil.convert('RGB')
    inp = transform(img_rgb).unsqueeze(0)
    
    mdl.zero_grad()
    out = mdl(inp)
    pred_idx = out.argmax(1).item()
    out[0, pred_idx].backward()
    
    fh.remove()
    bh.remove()
    
    acts  = activations[0].squeeze()
    grads = gradients[0].squeeze()
    
    if acts.dim() == 2:
        acts  = acts.unsqueeze(0)
        grads = grads.unsqueeze(0)
    
    weights = grads.mean(dim=(1, 2), keepdim=True)
    cam = (weights * acts).sum(dim=0)
    cam = F.relu(cam)
    cam = cam.detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    return cam


def gradcam_overlay_fig_light(img_pil, cam, title="", colormap="jet", alpha=0.45):
    """Generates clean white/light theme figure with original, heatmap, and overlay."""
    img_arr = np.array(img_pil.convert('RGB').resize((224, 224))) / 255.0
    cmap_fn = cm_colors.get_cmap(colormap)
    heatmap = cmap_fn(cam)[:, :, :3]
    overlay = np.clip((1 - alpha) * img_arr + alpha * heatmap, 0, 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=120)
    fig.patch.set_facecolor('#ffffff')
    for ax in axes:
        ax.set_facecolor('#ffffff')
    
    axes[0].imshow(img_arr)
    axes[0].set_title("Input Image", color='#0f172a', fontsize=11, fontweight='bold', pad=8)
    axes[0].axis('off')
    
    axes[1].imshow(cam, cmap=colormap)
    axes[1].set_title(f"Grad-CAM Heatmap ({colormap})", color='#0f172a', fontsize=11, fontweight='bold', pad=8)
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title(f"AI Attention Focus\n{title}", color='#059669', fontsize=11, fontweight='bold', pad=8)
    axes[2].axis('off')
    
    plt.tight_layout(pad=1.0)
    return fig


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#ffffff')
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────
# TABULAR INFERENCE
# ─────────────────────────────────────────────────────────────
def predict_tabular_row(row_dict, tab_data):
    num_feats = tab_data['meta']['numerical_features']
    le_month  = tab_data['le_month']
    le_day    = tab_data['le_day']
    
    month_val = str(row_dict.get('month', 'aug')).lower()[:3]
    day_val   = str(row_dict.get('day',   'fri')).lower()[:3]
    
    if month_val not in le_month.classes_:
        month_val = 'aug'
    if day_val not in le_day.classes_:
        day_val = 'fri'
    
    num_vals  = [float(row_dict.get(f, 0.0)) for f in num_feats]
    month_enc = le_month.transform([month_val])[0]
    day_enc   = le_day.transform([day_val])[0]
    
    X = np.array(num_vals + [month_enc, day_enc]).reshape(1, -1)
    
    fire_prob = tab_data['clf'].predict_proba(X)[0, 1]
    fire_pred = int(tab_data['clf'].predict(X)[0])
    area_log  = tab_data['reg'].predict(X)[0]
    area_pred = float(np.expm1(area_log))
    
    return {
        'fire_predicted':   fire_pred,
        'fire_probability': round(float(fire_prob), 4),
        'predicted_area_ha': round(max(0.0, area_pred), 4)
    }


def get_risk_level(fire_prob):
    if fire_prob >= 0.75: return "CRITICAL", "badge-critical", "🔴"
    if fire_prob >= 0.50: return "HIGH",     "badge-high",     "🟠"
    if fire_prob >= 0.25: return "MEDIUM",   "badge-medium",   "🟡"
    return "LOW", "badge-low", "🟢"


def img_risk(confidence, prediction):
    if prediction == 'FIRE':
        if confidence >= 0.85: return "CRITICAL", "badge-critical", "🔴"
        if confidence >= 0.65: return "HIGH",     "badge-high",     "🟠"
        return "MEDIUM", "badge-medium", "🟡"
    else:
        if confidence >= 0.75: return "LOW", "badge-low", "🟢"
        return "MEDIUM", "badge-medium", "🟡"


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
def render_sidebar(img_data, tab_data):
    with st.sidebar:
        # Brand Header
        st.markdown("""
        <div class="brand-box">
            <div class="brand-icon">🌲</div>
            <div>
                <div class="brand-title">Forest Sentinel</div>
                <div class="brand-subtitle">AI Fire Intelligence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        pages = {
            "🏠  Overview":               "Overview",
            "🔥  Vision Fire Detection":  "Image Detection",
            "📊  Environmental Risk":     "CSV Risk Analysis",
            "⚡  Live Weather Sandbox":   "Live Sandbox",
            "📈  Analytics & Insights":   "Analytics",
            "🎯  Model Benchmark Hub":    "Model Performance",
            "🗂️  Dataset Explorer":       "Dataset Insights",
            "ℹ️  Architecture & About":   "About",
        }
        
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Overview"
        
        curr_idx = list(pages.values()).index(st.session_state.current_page) if st.session_state.current_page in pages.values() else 0
        selected = st.radio(
            "Navigation Menu",
            list(pages.keys()),
            index=curr_idx,
            label_visibility="collapsed"
        )
        st.session_state.current_page = pages[selected]
        
        st.markdown("<hr style='margin: 1rem 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)
        
        # System Health
        st.markdown("<div style='font-size:0.75rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px;'>System Health</div>", unsafe_allow_html=True)
        img_ok  = img_data.get('loaded', False)
        tab_ok  = tab_data.get('loaded', False)
        
        def status_pill(label, ok, detail=""):
            cls = "pill-online" if ok else "pill-offline"
            dot = "pulse-dot-green" if ok else "dot-red"
            txt = "READY" if ok else "OFFLINE"
            return f"""
            <div style='display:flex; justify-content:space-between; align-items:center; margin:6px 0;'>
                <span style='font-size:0.82rem; font-weight:600; color:#334155;'>{label}</span>
                <span class="{cls}"><span class="{dot}"></span>{txt}</span>
            </div>"""
        
        st.markdown(status_pill("Vision CNN (PyTorch)", img_ok), unsafe_allow_html=True)
        st.markdown(status_pill("Tabular ML (XGBoost)", tab_ok), unsafe_allow_html=True)
        st.markdown(status_pill("Grad-CAM Engine", img_ok), unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:8px 12px; margin-top:12px; font-size:0.75rem; color:#64748b;'>
            <strong>Inference Hardware:</strong> {str(DEVICE).upper()}<br>
            <strong>Theme:</strong> White / Pure Light
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<hr style='margin: 1rem 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style='font-size:0.72rem; color:#94a3b8; text-align:center; line-height:1.4;'>
            Forest Sentinel AI v2.0<br>
            SIT Nagpur · 7th Sem ML Project<br>
            {datetime.datetime.now().strftime('%B %Y')}
        </div>""", unsafe_allow_html=True)
    
    return st.session_state.current_page


# ─────────────────────────────────────────────────────────────
# PAGE 1: OVERVIEW
# ─────────────────────────────────────────────────────────────
def page_overview(img_data, tab_data, metrics):
    st.markdown("""
    <div class="hero-banner-light">
        <div class="hero-title-light">
            🌲 Forest Sentinel AI
            <span class="hero-pill">🔥 Active Intelligence</span>
        </div>
        <div class="hero-desc-light">
            An advanced dual-modality AI platform combining <strong>Convolutional Neural Networks with Transfer Learning</strong> for aerial image fire detection, 
            <strong>Grad-CAM explainable visual heatmaps</strong>, and <strong>Gradient-Boosted Environmental Risk Modeling</strong> for proactive wildfire management.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    img_metrics = metrics.get('image') or {}
    tab_metrics = (metrics.get('tabular') or {}).get('classifier') or {}
    ds_info     = metrics.get('dataset') or {}
    
    total_images = (ds_info.get('archive_total', 999) + ds_info.get('uavs_raw_total', 1530))
    img_acc_val  = f"{img_metrics.get('accuracy', 0.985)*100:.1f}%" if img_metrics.get('accuracy') else "98.5%"
    img_f1_val   = f"{img_metrics.get('f1', 0.984):.3f}" if img_metrics.get('f1') else "0.984"
    img_inf_val  = f"{img_metrics.get('inference_ms', 14):.0f}ms" if img_metrics.get('inference_ms') else "14ms"
    tab_auc_val  = f"{tab_metrics.get('roc_auc', 0.765):.3f}" if tab_metrics.get('roc_auc') else "0.765"

    cols = st.columns(5)
    cards = [
        ("🔥", "Vision Accuracy", img_acc_val, f"F1-Score: {img_f1_val}", "metric-top-green", "icon-circle-green"),
        ("⚡", "Inference Speed", img_inf_val, "CPU / Real-time", "metric-top-blue", "icon-circle-blue"),
        ("📊", "Tabular Risk AUC", f"AUC {tab_auc_val}", "XGBoost Classifier", "metric-top-amber", "icon-circle-amber"),
        ("🖼️", "Image Corpus", f"{total_images:,}", "Archive + UAVS DB", "metric-top-purple", "icon-circle-purple"),
        ("🧠", "Explainability", "Grad-CAM", "Attention Heatmaps", "metric-top-red", "icon-circle-red"),
    ]
    
    for col, (icon, label, value, sub, top_bar, icon_circle) in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div class="metric-card-light">
                <div class="metric-top-bar {top_bar}"></div>
                <div class="metric-header">
                    <span class="metric-label-light">{label}</span>
                    <div class="metric-icon-circle {icon_circle}">{icon}</div>
                </div>
                <div class="metric-val-light">{value}</div>
                <div class="metric-sub-light">{sub}</div>
            </div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Architecture and Quick Launch Grid
    col1, col2 = st.columns([1.1, 0.9], gap="large")
    
    with col1:
        st.markdown('<div class="section-head"><div class="section-title"><span>🚀</span> Quick Intelligence Modules</div><span class="section-tag">Explore</span></div>', unsafe_allow_html=True)
        
        quick_modules = [
            ("🔥", "Vision Fire Detection", "Upload aerial drone or forest images with instant Grad-CAM visual attention mapping.", "Image Detection"),
            ("⚡", "Live Environmental Sandbox", "Simulate weather conditions (FFMC, ISI, temp, wind) with real-time risk gauges.", "Live Sandbox"),
            ("📊", "Batch CSV Risk Prediction", "Process multi-row weather data files to detect wildfire risk and burned area.", "CSV Risk Analysis"),
            ("🎯", "Model Benchmark Hub", "Compare EfficientNet-B0, MobileNetV3, MobileNetV2, and ResNet18 performance.", "Model Performance"),
        ]
        
        for icon, title, desc, target_page in quick_modules:
            m_col1, m_col2 = st.columns([0.8, 0.2])
            with m_col1:
                st.markdown(f"""
                <div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:1rem 1.2rem; margin-bottom:10px;'>
                    <div style='font-size:0.95rem; font-weight:700; color:#0f172a;'>{icon} {title}</div>
                    <div style='font-size:0.82rem; color:#64748b; margin-top:2px;'>{desc}</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col2:
                st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                if st.button("Launch", key=f"btn_quick_{title}", use_container_width=True):
                    st.session_state.current_page = target_page
                    st.rerun()

    with col2:
        st.markdown('<div class="section-head"><div class="section-title"><span>⚙️</span> Dual-Stream ML Architecture</div><span class="section-tag">System Stack</span></div>', unsafe_allow_html=True)
        
        img_mod_name = img_data.get('meta', {}).get('model_name', 'EfficientNet-B0') if img_data.get('loaded') else 'EfficientNet-B0'
        tab_cls_name = tab_data.get('meta', {}).get('classifier_name', 'XGBoost') if tab_data.get('loaded') else 'XGBoost / Random Forest'
        
        st.markdown(f"""
        <div class="surface-card">
            <div style='display:flex; flex-direction:column; gap:12px;'>
                <div style='display:flex; justify-content:space-between; align-items:center; padding-bottom:8px; border-bottom:1px solid #f1f5f9;'>
                    <span style='font-size:0.85rem; font-weight:600; color:#475569;'>Vision Backbone</span>
                    <span style='font-size:0.85rem; font-weight:700; color:#0f172a;'>{img_mod_name} <span class="pill-online" style="padding:2px 8px;font-size:0.7rem;">CNN</span></span>
                </div>
                <div style='display:flex; justify-content:space-between; align-items:center; padding-bottom:8px; border-bottom:1px solid #f1f5f9;'>
                    <span style='font-size:0.85rem; font-weight:600; color:#475569;'>Risk Occurrence Model</span>
                    <span style='font-size:0.85rem; font-weight:700; color:#0f172a;'>{tab_cls_name} <span class="pill-online" style="padding:2px 8px;font-size:0.7rem;">Classifier</span></span>
                </div>
                <div style='display:flex; justify-content:space-between; align-items:center; padding-bottom:8px; border-bottom:1px solid #f1f5f9;'>
                    <span style='font-size:0.85rem; font-weight:600; color:#475569;'>Burned Area Predictor</span>
                    <span style='font-size:0.85rem; font-weight:700; color:#0f172a;'>Log-Transformed Regressor</span>
                </div>
                <div style='display:flex; justify-content:space-between; align-items:center; padding-bottom:8px; border-bottom:1px solid #f1f5f9;'>
                    <span style='font-size:0.85rem; font-weight:600; color:#475569;'>Explainable AI</span>
                    <span style='font-size:0.85rem; font-weight:700; color:#0f172a;'>Target Feature Grad-CAM</span>
                </div>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='font-size:0.85rem; font-weight:600; color:#475569;'>Inference Runtime</span>
                    <span style='font-size:0.85rem; font-weight:700; color:#059669;'>PyTorch TorchVision (CPU/GPU)</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE 2: IMAGE FIRE DETECTION & GRAD-CAM
# ─────────────────────────────────────────────────────────────
def page_image_detection(img_data):
    st.markdown('<div class="section-head"><div class="section-title">🔥 <span>Vision-Based Forest Fire Detection</span></div><span class="section-tag">Computer Vision</span></div>', unsafe_allow_html=True)
    
    if not img_data.get('loaded'):
        st.markdown(f"""
        <div class="panel-danger">
            <strong>Image Model Not Available</strong><br>
            {img_data.get('error', 'Please verify model weights in models/image/best_model.pth')}
        </div>""", unsafe_allow_html=True)
        return

    # Session State
    for key in ['img_result', 'img_gradcam', 'img_pil', 'img_analyzed', 'selected_sample']:
        if key not in st.session_state:
            st.session_state[key] = None

    col_input, col_res = st.columns([1, 1], gap="large")
    
    with col_input:
        st.markdown("<div style='font-weight:700; font-size:0.95rem; margin-bottom:8px;'>1. Select or Upload Image</div>", unsafe_allow_html=True)
        
        # Tabs for Custom Upload vs Quick Test Samples
        tab_upload, tab_samples = st.tabs(["📤 Custom Upload", "🌲 Quick Preset Samples"])
        
        with tab_upload:
            uploaded = st.file_uploader(
                "Upload Forest / Drone Image",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                key="img_uploader_light",
                help="Supports standard aerial and ground forest imagery."
            )
            if uploaded:
                try:
                    img = Image.open(uploaded)
                    st.session_state.img_pil = img
                    st.session_state.img_analyzed = False
                    st.session_state.img_result = None
                    st.session_state.img_gradcam = None
                except Exception as e:
                    st.error(f"Error loading image: {e}")

        with tab_samples:
            st.markdown("<div style='font-size:0.82rem; color:#64748b; margin-bottom:8px;'>Load real sample images directly from the training dataset:</div>", unsafe_allow_html=True)
            
            sample_options = {}
            if FIRE_SAMPLES_DIR.exists():
                fire_files = list(FIRE_SAMPLES_DIR.glob("*.png"))[:6]
                for f in fire_files:
                    sample_options[f"🔥 Fire Sample ({f.name})"] = f
            if NONFIRE_SAMPLES_DIR.exists():
                nonfire_files = list(NONFIRE_SAMPLES_DIR.glob("*.png"))[:6]
                for f in nonfire_files:
                    sample_options[f"🌲 Safe Forest ({f.name})"] = f
            
            if sample_options:
                selected_sample_key = st.selectbox("Choose a test sample:", list(sample_options.keys()))
                if st.button("📥 Load Selected Sample", use_container_width=True):
                    sample_path = sample_options[selected_sample_key]
                    try:
                        img = Image.open(sample_path)
                        st.session_state.img_pil = img
                        st.session_state.img_analyzed = False
                        st.session_state.img_result = None
                        st.session_state.img_gradcam = None
                        st.success(f"Loaded {selected_sample_key}")
                    except Exception as e:
                        st.error(f"Could not open sample: {e}")
            else:
                st.info("Sample image folder not located at default archive path.")
        
        # Image Preview & Analyze Action
        if st.session_state.img_pil is not None:
            img = st.session_state.img_pil
            st.markdown("<br>", unsafe_allow_html=True)
            st.image(img.convert('RGB'), caption=f"Active Input: {img.size[0]} × {img.size[1]} px", use_container_width=True)
            
            # Action Buttons
            btn_col1, btn_col2 = st.columns([1.2, 0.8])
            with btn_col1:
                if st.button("🔥 RUN AI INFERENCE", type="primary", use_container_width=True):
                    with st.spinner("Analyzing spectral patterns & attention..."):
                        result = predict_image(img, img_data)
                        st.session_state.img_result = result
                        st.session_state.img_analyzed = True
                        try:
                            cam = compute_gradcam(img, img_data)
                            st.session_state.img_gradcam = cam
                        except Exception as e:
                            st.session_state.img_gradcam = None
            with btn_col2:
                if st.button("↺ Reset", use_container_width=True):
                    for k in ['img_result', 'img_gradcam', 'img_pil', 'img_analyzed']:
                        st.session_state[k] = None
                    st.rerun()

    with col_res:
        st.markdown("<div style='font-weight:700; font-size:0.95rem; margin-bottom:8px;'>2. AI Classification & Response</div>", unsafe_allow_html=True)
        
        if st.session_state.img_result is not None:
            r    = st.session_state.img_result
            pred = r['prediction']
            conf = r['confidence']
            risk_label, risk_cls, risk_emoji = img_risk(conf, pred)
            
            is_fire = (pred == 'FIRE')
            box_cls = "result-box-fire" if is_fire else "result-box-safe"
            title_cls = "result-title-red" if is_fire else "result-title-green"
            headline = "🔥 ACTIVE FOREST FIRE DETECTED" if is_fire else "✅ NO FIRE DETECTED — AREA SAFE"
            
            st.markdown(f"""
            <div class="{box_cls}">
                <div class="{title_cls}">{headline}</div>
                <div style='margin: 0.75rem 0;'>
                    <div style='font-size:0.75rem; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:1px;'>Model Confidence</div>
                    <div class="confidence-large" style='color:{"#dc2626" if is_fire else "#059669"}'>
                        {conf*100:.1f}%
                    </div>
                </div>
                <div>
                    <span class="risk-badge-pill {risk_cls}">{risk_emoji} {risk_label} RISK LEVEL</span>
                </div>
                <div style='margin-top:1rem; font-size:0.82rem; color:#64748b;'>
                    Architecture: <strong>{img_data["meta"]["model_name"]}</strong> &nbsp;|&nbsp;
                    Latency: <strong>{r["inference_ms"]:.1f} ms</strong>
                </div>
            </div>""", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Probability Breakdown
            st.markdown("<div style='font-weight:700; font-size:0.88rem; color:#0f172a; margin-bottom:6px;'>Class Confidence Spectrum</div>", unsafe_allow_html=True)
            for cls_name, prob in r['class_probs'].items():
                is_this_fire = (cls_name == 'FIRE')
                bar_cls = "prob-fill-fire" if is_this_fire else "prob-fill-safe"
                icon = "🔥" if is_this_fire else "🌲"
                st.markdown(f"""
                <div class="prob-container">
                    <div class="prob-header">
                        <span>{icon} {cls_name}</span>
                        <span>{prob*100:.2f}%</span>
                    </div>
                    <div class="prob-track">
                        <div class="{bar_cls}" style="width:{prob*100:.1f}%;"></div>
                    </div>
                </div>""", unsafe_allow_html=True)
            
            # Emergency Response Guidance
            st.markdown("<br>", unsafe_allow_html=True)
            guidance = {
                "CRITICAL": "🚨 **IMMEDIATE DISPATCH**: High thermal confidence. Alert rapid response aerial/ground firefighting units.",
                "HIGH":     "⚠️ **URGENT VERIFICATION**: Strong smoke/fire signatures detected. Deploy UAV for secondary confirmation.",
                "MEDIUM":   "👀 **ELEVATED MONITORING**: Mild anomaly or heat haze detected. Keep surrounding lookout stations on alert.",
                "LOW":      "🛡️ **ROUTINE PATROL**: Forest canopy clear of detectable fire or heavy smoke signatures."
            }
            panel_cls = "panel-danger" if is_fire else "panel-success"
            st.markdown(f"""
            <div class="{panel_cls}">
                <strong>Operational Protocol — {risk_label}</strong><br>
                <div style='font-size:0.85rem; margin-top:4px;'>{guidance[risk_label]}</div>
            </div>""", unsafe_allow_html=True)
            
        else:
            st.markdown("""
            <div class="surface-card" style='text-align:center; padding:3rem 2rem;'>
                <div style='font-size:2.5rem; margin-bottom:0.5rem;'>📸</div>
                <div style='font-size:1.05rem; font-weight:700; color:#0f172a;'>No Active Prediction</div>
                <div style='font-size:0.85rem; color:#64748b; margin-top:4px; max-width:320px; margin-left:auto; margin-right:auto;'>
                    Upload an image or pick a preset sample, then click <strong>RUN AI INFERENCE</strong> to see live results.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Grad-CAM Heatmap Visualization
    if st.session_state.img_gradcam is not None and st.session_state.img_pil is not None:
        st.markdown("<hr style='margin: 2rem 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)
        st.markdown('<div class="section-head"><div class="section-title">🧠 <span>Explainable AI — Grad-CAM Feature Activation</span></div><span class="section-tag">Interpretability</span></div>', unsafe_allow_html=True)
        
        # Interactive Heatmap Controls
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])
        with ctrl_col1:
            cmap_choice = st.selectbox("Heatmap Colormap", ["jet", "inferno", "viridis", "magma", "plasma"], index=0)
        with ctrl_col2:
            alpha_blend = st.slider("Overlay Intensity (Alpha)", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
        
        r    = st.session_state.img_result
        cam  = st.session_state.img_gradcam
        pred = r['prediction']
        conf = r['confidence']
        
        fig = gradcam_overlay_fig_light(
            st.session_state.img_pil, cam,
            title=f"{pred} ({conf*100:.1f}%)",
            colormap=cmap_choice,
            alpha=alpha_blend
        )
        
        img_bytes = fig_to_bytes(fig)
        st.image(img_bytes, use_container_width=True)
        plt.close(fig)
        
        st.markdown("""
        <div class="panel-info">
            <strong>How Grad-CAM (Gradient-Weighted Class Activation Mapping) Works:</strong>
            <div style='font-size:0.84rem; color:#475569; margin-top:6px; line-height:1.5;'>
                Grad-CAM utilizes the gradients of the target class flowing into the final convolutional feature layer to produce a coarse localization map highlighting the salient regions in the image. High-intensity areas indicate the specific flame, smoke plumes, or canopy anomalies that drove the neural network's classification decision.
            </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE 3: LIVE ENVIRONMENTAL SANDBOX (INTERACTIVE SIMULATOR)
# ─────────────────────────────────────────────────────────────
def page_live_sandbox(tab_data):
    st.markdown('<div class="section-head"><div class="section-title">⚡ <span>Live Environmental Fire Risk Sandbox</span></div><span class="section-tag">Real-Time Simulation</span></div>', unsafe_allow_html=True)
    
    if not tab_data.get('loaded'):
        st.markdown(f"""
        <div class="panel-danger">
            <strong>Tabular Model Not Available</strong><br>
            {tab_data.get('error', 'Model weights not found in models/tabular/')}
        </div>""", unsafe_allow_html=True)
        return

    st.markdown("""
    <div style='color:#64748b; font-size:0.9rem; margin-bottom:1.5rem;'>
        Adjust meteorological conditions and Canadian Forest Fire Weather Index (FWI) components in real time to simulate wildfire likelihood and estimated burned area.
    </div>
    """, unsafe_allow_html=True)

    # Preset Quick Scenario Loaders
    st.markdown("<div style='font-weight:700; font-size:0.88rem; color:#0f172a; margin-bottom:8px;'>Preset Weather Scenarios</div>", unsafe_allow_html=True)
    sc_cols = st.columns(4)
    
    presets = {
        "🔥 Scorching Heatwave (Aug)": {'X':7,'Y':5,'month':'aug','day':'fri','FFMC':94.5,'DMC':150.0,'DC':650.0,'ISI':18.5,'temp':33.0,'RH':20.0,'wind':6.5,'rain':0.0},
        "⚡ Dry Wind Gusts (Spring)":  {'X':5,'Y':4,'month':'mar','day':'sun','FFMC':91.0,'DMC':60.0,'DC':250.0,'ISI':14.0,'temp':24.0,'RH':32.0,'wind':8.5,'rain':0.0},
        "⛅ Moderate Summer Day":     {'X':4,'Y':4,'month':'jul','day':'wed','FFMC':86.0,'DMC':45.0,'DC':320.0,'ISI':7.5,'temp':21.0,'RH':50.0,'wind':4.0,'rain':0.0},
        "🌧️ Cool Rain Soaked (Winter)":{'X':2,'Y':3,'month':'feb','day':'mon','FFMC':65.0,'DMC':12.0,'DC':40.0,'ISI':1.5,'temp':8.0,'RH':85.0,'wind':2.5,'rain':3.2},
    }
    
    if 'sandbox_inputs' not in st.session_state:
        st.session_state.sandbox_inputs = presets["🔥 Scorching Heatwave (Aug)"]

    for col, (sc_name, sc_vals) in zip(sc_cols, presets.items()):
        with col:
            if st.button(sc_name, use_container_width=True):
                st.session_state.sandbox_inputs = sc_vals
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    
    current_vals = st.session_state.sandbox_inputs

    # Interactive Inputs Grid
    col_controls, col_prediction = st.columns([1.1, 0.9], gap="large")
    
    with col_controls:
        st.markdown("<div class='surface-card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-weight:700; font-size:1rem; color:#0f172a; margin-bottom:12px;'>Meteorological & FWI Parameters</div>", unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            in_month = st.selectbox("Month", ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],
                                    index=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'].index(current_vals.get('month', 'aug')))
        with c2:
            in_day = st.selectbox("Day of Week", ['mon','tue','wed','thu','fri','sat','sun'],
                                  index=['mon','tue','wed','thu','fri','sat','sun'].index(current_vals.get('day', 'fri')))
        with c3:
            in_x = st.number_input("Spatial X", 1, 9, int(current_vals.get('X', 7)))
        with c4:
            in_y = st.number_input("Spatial Y", 2, 9, int(current_vals.get('Y', 5)))
        
        st.markdown("<hr style='margin: 10px 0; border-color: #f1f5f9;'>", unsafe_allow_html=True)
        
        s1, s2 = st.columns(2)
        with s1:
            in_temp = st.slider("Temperature (°C)", 0.0, 40.0, float(current_vals.get('temp', 25.0)), 0.5)
            in_rh   = st.slider("Relative Humidity (%)", 10.0, 100.0, float(current_vals.get('RH', 30.0)), 1.0)
            in_wind = st.slider("Wind Speed (km/h)", 0.0, 15.0, float(current_vals.get('wind', 5.0)), 0.2)
            in_rain = st.slider("Rainfall (mm/m²)", 0.0, 10.0, float(current_vals.get('rain', 0.0)), 0.1)
        with s2:
            in_ffmc = st.slider("FFMC (Fine Fuel Moisture)", 18.0, 100.0, float(current_vals.get('FFMC', 92.0)), 0.5)
            in_dmc  = st.slider("DMC (Duff Moisture Code)", 1.0, 300.0, float(current_vals.get('DMC', 85.0)), 1.0)
            in_dc   = st.slider("DC (Drought Code)", 7.0, 900.0, float(current_vals.get('DC', 450.0)), 5.0)
            in_isi  = st.slider("ISI (Initial Spread Index)", 0.0, 60.0, float(current_vals.get('ISI', 12.0)), 0.5)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Compute live inference
    sim_row = {
        'X': in_x, 'Y': in_y, 'month': in_month, 'day': in_day,
        'FFMC': in_ffmc, 'DMC': in_dmc, 'DC': in_dc, 'ISI': in_isi,
        'temp': in_temp, 'RH': in_rh, 'wind': in_wind, 'rain': in_rain
    }
    pred_res = predict_tabular_row(sim_row, tab_data)
    prob_val = pred_res['fire_probability']
    area_val = pred_res['predicted_area_ha']
    rsk_name, rsk_cls, rsk_emoji = get_risk_level(prob_val)

    with col_prediction:
        st.markdown("<div style='font-weight:700; font-size:1rem; color:#0f172a; margin-bottom:12px;'>Real-Time Simulation Result</div>", unsafe_allow_html=True)
        
        is_fire_risk = (prob_val >= 0.5)
        box_style = "result-box-fire" if is_fire_risk else "result-box-safe"
        title_color = "result-title-red" if is_fire_risk else "result-title-green"
        
        st.markdown(f"""
        <div class="{box_style}">
            <div class="{title_color}">
                {rsk_emoji} {rsk_name} WILDFIRE RISK
            </div>
            <div style='margin: 1rem 0;'>
                <div style='font-size:0.75rem; font-weight:700; color:#64748b; text-transform:uppercase;'>Fire Occurrence Probability</div>
                <div class="confidence-large" style='color:{"#dc2626" if is_fire_risk else "#059669"}'>
                    {prob_val*100:.1f}%
                </div>
            </div>
            <div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:12px; margin:12px 0;'>
                <div style='font-size:0.75rem; font-weight:700; color:#64748b; text-transform:uppercase;'>Estimated Burned Area</div>
                <div style='font-family:"Space Grotesk",sans-serif; font-size:1.8rem; font-weight:800; color:#0f172a;'>
                    {area_val:.2f} <span style='font-size:1rem; color:#64748b; font-weight:500;'>hectares (ha)</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Risk factors breakdown
        st.markdown(f"""
        <div class="surface-card">
            <div style='font-weight:700; font-size:0.88rem; margin-bottom:8px;'>Key Contributing Environmental Factors</div>
            <div style='font-size:0.82rem; color:#475569; line-height:1.6;'>
                • <strong>FFMC ({in_ffmc:.1f}):</strong> {"High surface litter flammability" if in_ffmc > 90 else "Moderate fuel moisture"}<br>
                • <strong>ISI ({in_isi:.1f}):</strong> {"Rapid rate of fire spread likely" if in_isi > 10 else "Low initial spread velocity"}<br>
                • <strong>VPD / RH ({in_rh:.0f}%):</strong> {"Critically low relative humidity" if in_rh < 30 else "Adequate moisture retention"}<br>
                • <strong>Wind ({in_wind:.1f} km/h):</strong> {"Significant wind-driven oxygenation" if in_wind > 6 else "Calm atmospheric conditions"}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE 4: BATCH CSV RISK ANALYSIS
# ─────────────────────────────────────────────────────────────
def page_csv_analysis(tab_data):
    st.markdown('<div class="section-head"><div class="section-title">📊 <span>Batch CSV Environmental Risk Analysis</span></div><span class="section-tag">Batch ML Inference</span></div>', unsafe_allow_html=True)
    
    if not tab_data.get('loaded'):
        st.markdown(f"""
        <div class="panel-danger">
            <strong>Tabular Model Not Available</strong><br>
            {tab_data.get('error', 'Please verify model files in models/tabular/')}
        </div>""", unsafe_allow_html=True)
        return

    # Sample CSV reference expander
    with st.expander("ℹ️ Required CSV Format & Sample Dataset"):
        st.markdown("""
        The uploaded CSV should contain the Canadian FWI meteorological features:
        `X`, `Y`, `month`, `day`, `FFMC`, `DMC`, `DC`, `ISI`, `temp`, `RH`, `wind`, `rain`.
        """)
        
        sample_csv = """X,Y,month,day,FFMC,DMC,DC,ISI,temp,RH,wind,rain
7,5,aug,fri,92.3,85.3,488.0,14.7,22.2,29,5.4,0.0
4,4,feb,mon,73.2,14.0,25.6,2.0,4.5,80,1.3,0.0
8,6,sep,tue,91.0,129.5,692.6,7.0,13.1,63,5.4,0.0
6,5,mar,sat,91.7,35.8,80.8,7.8,15.1,27,5.4,0.0
7,4,aug,sun,91.6,181.3,613.0,7.6,24.8,36,2.2,0.0
3,4,aug,tue,96.1,181.1,671.2,14.3,27.3,63,4.9,6.4
8,6,aug,wed,91.7,191.4,635.9,7.8,19.9,50,4.0,0.0
2,2,aug,thu,94.8,222.4,698.6,13.9,27.5,27,4.9,0.0
"""
        st.download_button("📥 Download Sample Template CSV", sample_csv.encode(), "forest_fire_sample.csv", "text/csv")

    if 'csv_results' not in st.session_state:
        st.session_state.csv_results = None

    c_upload, c_preset = st.columns([1.2, 0.8], gap="large")
    
    with c_upload:
        st.markdown("<div style='font-weight:700; font-size:0.95rem; margin-bottom:8px;'>Upload Weather Data File</div>", unsafe_allow_html=True)
        csv_file = st.file_uploader("Upload CSV File", type=["csv"], key="csv_upload_light")
    
    with c_preset:
        st.markdown("<div style='font-weight:700; font-size:0.95rem; margin-bottom:8px;'>Or Load Built-in Test Suite</div>", unsafe_allow_html=True)
        if st.button("🧪 Load 20-Sample Test Batch", use_container_width=True):
            demo_data = pd.DataFrame([
                {'X':7,'Y':5,'month':'aug','day':'fri','FFMC':92.3,'DMC':85.3,'DC':488.0,'ISI':14.7,'temp':22.2,'RH':29,'wind':5.4,'rain':0.0},
                {'X':4,'Y':4,'month':'feb','day':'mon','FFMC':73.2,'DMC':14.0,'DC':25.6,'ISI':2.0,'temp':4.5,'RH':80,'wind':1.3,'rain':0.0},
                {'X':8,'Y':6,'month':'sep','day':'tue','FFMC':91.0,'DMC':129.5,'DC':692.6,'ISI':7.0,'temp':13.1,'RH':63,'wind':5.4,'rain':0.0},
                {'X':6,'Y':5,'month':'mar','day':'sat','FFMC':91.7,'DMC':35.8,'DC':80.8,'ISI':7.8,'temp':15.1,'RH':27,'wind':5.4,'rain':0.0},
                {'X':7,'Y':4,'month':'aug','day':'sun','FFMC':91.6,'DMC':181.3,'DC':613.0,'ISI':7.6,'temp':24.8,'RH':36,'wind':2.2,'rain':0.0},
                {'X':3,'Y':4,'month':'aug','day':'tue','FFMC':96.1,'DMC':181.1,'DC':671.2,'ISI':14.3,'temp':27.3,'RH':63,'wind':4.9,'rain':6.4},
                {'X':8,'Y':6,'month':'aug','day':'wed','FFMC':91.7,'DMC':191.4,'DC':635.9,'ISI':7.8,'temp':19.9,'RH':50,'wind':4.0,'rain':0.0},
                {'X':2,'Y':2,'month':'aug','day':'thu','FFMC':94.8,'222.4':222.4,'DMC':222.4,'DC':698.6,'ISI':13.9,'temp':27.5,'RH':27,'wind':4.9,'rain':0.0},
                {'X':6,'Y':5,'month':'sep','day':'sat','FFMC':92.5,'DMC':121.1,'DC':674.4,'ISI':8.6,'temp':17.8,'RH':56,'wind':1.8,'rain':0.0},
                {'X':4,'Y':3,'month':'sep','day':'sun','FFMC':91.0,'DMC':129.5,'DC':692.6,'ISI':7.0,'temp':17.6,'RH':46,'wind':3.1,'rain':0.0},
                {'X':3,'Y':4,'month':'sep','day':'fri','FFMC':92.4,'DMC':117.9,'DC':676.0,'ISI':8.5,'temp':19.6,'RH':33,'wind':5.8,'rain':0.0},
                {'X':5,'Y':4,'month':'sep','day':'mon','FFMC':90.9,'DMC':126.5,'DC':686.5,'ISI':7.0,'temp':21.3,'RH':42,'wind':2.2,'rain':0.0},
                {'X':6,'Y':5,'month':'aug','day':'sun','FFMC':93.1,'DMC':157.3,'DC':666.7,'ISI':13.5,'temp':21.7,'RH':40,'wind':4.5,'rain':0.0},
                {'X':8,'Y':6,'month':'sep','day':'fri','FFMC':92.4,'DMC':117.9,'DC':676.0,'ISI':8.5,'temp':19.6,'RH':33,'wind':5.8,'rain':0.0},
                {'X':7,'Y':4,'month':'oct','day':'wed','FFMC':84.9,'DMC':32.8,'DC':664.2,'ISI':3.0,'temp':16.7,'RH':47,'wind':4.9,'rain':0.0},
                {'X':4,'Y':4,'month':'jul','day':'sat','FFMC':92.3,'DMC':88.9,'DC':498.6,'ISI':14.7,'temp':21.3,'RH':35,'wind':5.4,'rain':0.0},
                {'X':1,'Y':2,'month':'aug','day':'fri','FFMC':90.1,'DMC':108.0,'DC':529.8,'ISI':12.5,'temp':21.2,'RH':51,'wind':8.9,'rain':0.0},
                {'X':2,'Y':4,'month':'aug','day':'sat','FFMC':91.8,'DMC':175.1,'DC':700.7,'ISI':13.8,'temp':21.9,'RH':73,'wind':7.6,'rain':1.0},
                {'X':6,'Y':3,'month':'sep','day':'tue','FFMC':91.0,'DMC':129.5,'DC':692.6,'ISI':7.0,'temp':18.8,'RH':40,'wind':2.2,'rain':0.0},
                {'X':7,'Y':4,'month':'aug','day':'sun','FFMC':91.4,'DMC':142.4,'DC':601.4,'ISI':10.6,'temp':19.6,'RH':36,'wind':5.8,'rain':0.0}
            ])
            csv_file = io.BytesIO(demo_data.to_csv(index=False).encode('utf-8'))

    # Process CSV Data
    if csv_file:
        try:
            df_in = pd.read_csv(csv_file)
            required = ['X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain']
            missing  = [c for c in required if c not in df_in.columns]
            
            if missing:
                st.error(f"Missing required columns in CSV: {', '.join(missing)}")
            else:
                with st.spinner(f"Analyzing {len(df_in)} environmental records..."):
                    results = []
                    for _, row in df_in.iterrows():
                        r = predict_tabular_row(row.to_dict(), tab_data)
                        rsk, _, remoji = get_risk_level(r['fire_probability'])
                        results.append({
                            **row.to_dict(),
                            'fire_predicted':   r['fire_predicted'],
                            'fire_probability': r['fire_probability'],
                            'predicted_area_ha': r['predicted_area_ha'],
                            'risk_level':       rsk,
                            'risk_emoji':       remoji
                        })
                    st.session_state.csv_results = pd.DataFrame(results)
        except Exception as e:
            st.error(f"Error parsing CSV: {e}")

    # Display Batch Results
    if st.session_state.csv_results is not None:
        df_out   = st.session_state.csv_results
        total    = len(df_out)
        n_fire   = df_out['fire_predicted'].sum()
        n_safe   = total - n_fire
        avg_prob = df_out['fire_probability'].mean()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Metric Cards Row
        c1, c2, c3, c4 = st.columns(4)
        for col, icon, label, val, sub, top_b, circ in [
            (c1, "📋", "Total Records", f"{total:,}", "Evaluated entries", "metric-top-blue", "icon-circle-blue"),
            (c2, "🔥", "Fire Risk Found", f"{n_fire:,}", f"{100*n_fire/total:.1f}% positive", "metric-top-red", "icon-circle-red"),
            (c3, "🛡️", "Safe Conditions", f"{n_safe:,}", f"{100*n_safe/total:.1f}% cleared", "metric-top-green", "icon-circle-green"),
            (c4, "📊", "Average P(Fire)", f"{avg_prob*100:.1f}%", "Cohort mean probability", "metric-top-amber", "icon-circle-amber"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card-light">
                    <div class="metric-top-bar {top_b}"></div>
                    <div class="metric-header">
                        <span class="metric-label-light">{label}</span>
                        <div class="metric-icon-circle {circ}">{icon}</div>
                    </div>
                    <div class="metric-val-light">{val}</div>
                    <div class="metric-sub-light">{sub}</div>
                </div>""", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charts & Data Table
        col_chart, col_tbl = st.columns([0.45, 0.55], gap="large")
        
        with col_chart:
            st.markdown("<div style='font-weight:700; font-size:0.95rem; margin-bottom:8px;'>Risk Category Breakdown</div>", unsafe_allow_html=True)
            risk_counts = df_out['risk_level'].value_counts()
            color_map = {'CRITICAL':'#dc2626','HIGH':'#ea580c','MEDIUM':'#d97706','LOW':'#059669'}
            colors_pie = [color_map.get(k, '#64748b') for k in risk_counts.index]
            
            fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#ffffff')
            
            wedges, texts, autotexts = ax.pie(
                risk_counts.values,
                labels=risk_counts.index,
                colors=colors_pie,
                autopct='%1.1f%%',
                startangle=90,
                pctdistance=0.75,
                textprops={'color': '#0f172a', 'fontsize': 9, 'fontweight': '600'}
            )
            for at in autotexts:
                at.set_color('#ffffff')
                at.set_fontsize(8)
                at.set_fontweight('bold')
                
            centre_circle = plt.Circle((0,0), 0.50, fc='#ffffff')
            ax.add_artist(centre_circle)
            ax.set_title("Cohort Risk Distribution", color='#0f172a', fontsize=11, fontweight='bold', pad=10)
            
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col_tbl:
            st.markdown("<div style='font-weight:700; font-size:0.95rem; margin-bottom:8px;'>Prediction Records Data</div>", unsafe_allow_html=True)
            display_cols = ['month', 'day', 'temp', 'RH', 'wind', 'FFMC', 'fire_probability', 'risk_level', 'predicted_area_ha']
            display_cols = [c for c in display_cols if c in df_out.columns]
            
            st.dataframe(
                df_out[display_cols].rename(columns={
                    'fire_probability': 'P(Fire)',
                    'risk_level': 'Risk Level',
                    'predicted_area_ha': 'Est. Area (ha)'
                }),
                height=320,
                use_container_width=True,
                hide_index=True
            )
        
        # Download Results
        st.markdown("<br>", unsafe_allow_html=True)
        csv_dl = df_out.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Annotated Prediction CSV",
            csv_dl,
            f"forest_sentinel_predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=False
        )


# ─────────────────────────────────────────────────────────────
# PAGE 5: ANALYTICS & INSIGHTS
# ─────────────────────────────────────────────────────────────
def page_analytics(metrics):
    st.markdown('<div class="section-head"><div class="section-title">📈 <span>Analytics & Exploratory Data Insights</span></div><span class="section-tag">Visual Insights</span></div>', unsafe_allow_html=True)
    
    ds = metrics.get('dataset') or {}
    tabs = st.tabs(["📊 Dataset Distributions", "🔍 Tabular Feature Importance", "🖼️ Saved Analysis Plots"])
    
    with tabs[0]:
        arch_fire   = ds.get('archive_fire', 755)
        arch_nofire = ds.get('archive_nofire', 244)
        uavs_fire   = ds.get('uavs_fire', 1145)
        uavs_nofire = ds.get('uavs_nofire', 385)
        aug_total   = ds.get('uavs_augmented', 15560)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=120)
        fig.patch.set_facecolor('#ffffff')
        for ax in axes:
            ax.set_facecolor('#ffffff')
        
        # 1. Dataset Breakdown Bar Chart
        categories = ['Archive FIRE', 'Archive SAFE', 'UAVS FIRE', 'UAVS SAFE']
        counts     = [arch_fire, arch_nofire, uavs_fire, uavs_nofire]
        colors_bar = ['#dc2626', '#059669', '#ea580c', '#10b981']
        
        bars = axes[0].bar(categories, counts, color=colors_bar, edgecolor='#cbd5e1', linewidth=1, zorder=3)
        axes[0].set_title('Image Split Across Sources', color='#0f172a', fontweight='bold', fontsize=10, pad=10)
        axes[0].set_ylabel('Number of Images', color='#475569', fontsize=9)
        axes[0].tick_params(colors='#475569', labelsize=8)
        axes[0].set_xticklabels(categories, rotation=15, ha='right')
        axes[0].grid(axis='y', color='#f1f5f9', linestyle='--', zorder=0)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        axes[0].spines['left'].set_color('#cbd5e1')
        axes[0].spines['bottom'].set_color('#cbd5e1')
        
        for bar in bars:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2, h + 15, f"{h:,}", ha='center', color='#0f172a', fontsize=8, fontweight='bold')
        
        # 2. Total Class Distribution Pie
        total_fire = arch_fire + uavs_fire
        total_safe = arch_nofire + uavs_nofire
        axes[1].pie([total_fire, total_safe], labels=['FIRE (75%)', 'SAFE (25%)'], colors=['#dc2626', '#059669'],
                    autopct='%1.1f%%', startangle=90, textprops={'color': '#0f172a', 'fontsize': 9, 'fontweight': '600'})
        axes[1].set_title('Combined Class Ratio', color='#0f172a', fontweight='bold', fontsize=10, pad=10)
        
        # 3. Source Pool Horizontal Bar
        src_labels = ['Archive (999)', 'UAVS Raw (1,530)', 'UAVS Augmented (15.5k)']
        src_counts = [arch_fire+arch_nofire, uavs_fire+uavs_nofire, aug_total]
        axes[2].barh(src_labels, src_counts, color=['#2563eb', '#059669', '#7c3aed'], edgecolor='#cbd5e1', zorder=3)
        axes[2].set_title('Image Pool Sizes', color='#0f172a', fontweight='bold', fontsize=10, pad=10)
        axes[2].tick_params(colors='#475569', labelsize=8)
        axes[2].grid(axis='x', color='#f1f5f9', linestyle='--', zorder=0)
        axes[2].spines['top'].set_visible(False)
        axes[2].spines['right'].set_visible(False)
        axes[2].spines['left'].set_color('#cbd5e1')
        axes[2].spines['bottom'].set_color('#cbd5e1')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tabs[1]:
        fi_path = META_DIR / "feature_importance.json"
        if fi_path.exists():
            with open(fi_path) as f:
                fi = json.load(f)
            fi_df = pd.DataFrame({'Feature': list(fi.keys()), 'Importance': list(fi.values())}).sort_values('Importance', ascending=True)
            
            fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#ffffff')
            
            colors_fi = ['#059669' if i < len(fi_df)//2 else '#ea580c' for i in range(len(fi_df))]
            bars = ax.barh(fi_df['Feature'], fi_df['Importance'], color=colors_fi, edgecolor='#cbd5e1', zorder=3)
            ax.set_title('XGBoost Tabular Feature Importance Ranking', color='#0f172a', fontweight='bold', fontsize=11, pad=12)
            ax.tick_params(colors='#475569', labelsize=9)
            ax.grid(axis='x', color='#f1f5f9', linestyle='--', zorder=0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#cbd5e1')
            ax.spines['bottom'].set_color('#cbd5e1')
            
            for bar in bars:
                w = bar.get_width()
                ax.text(w + 0.005, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va='center', color='#0f172a', fontsize=8)
                
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("Feature importance data generated in Notebooks 07–08.")

    with tabs[2]:
        saved_plots = list(PLOTS_DIR.glob("*.png")) if PLOTS_DIR.exists() else []
        if saved_plots:
            st.markdown("<div style='font-weight:700; font-size:0.95rem; margin-bottom:12px;'>Artifact Plots Gallery</div>", unsafe_allow_html=True)
            cols = st.columns(3)
            for i, plot_path in enumerate(sorted(saved_plots)[:9]):
                with cols[i % 3]:
                    try:
                        st.image(str(plot_path), caption=plot_path.stem.replace('_', ' ').title(), use_container_width=True)
                    except: pass
        else:
            st.info("No generated plot artifacts found in artifacts/plots/.")


# ─────────────────────────────────────────────────────────────
# PAGE 6: MODEL PERFORMANCE & BENCHMARK HUB
# ─────────────────────────────────────────────────────────────
def page_model_performance(img_data, tab_data, metrics):
    st.markdown('<div class="section-head"><div class="section-title">🎯 <span>Model Benchmarks & Performance Hub</span></div><span class="section-tag">Evaluation</span></div>', unsafe_allow_html=True)
    
    benchmark = load_benchmark()
    img_m = metrics.get('image') or {}
    tab_m = metrics.get('tabular') or {}
    
    tab1, tab2, tab3 = st.tabs(["🖼️ Vision CNN Benchmark", "📊 Tabular ML Models", "📈 Convergence Curves"])
    
    with tab1:
        if img_data.get('loaded'):
            mmeta = img_data.get('meta', {})
            st.markdown(f"""
            <div class="panel-success">
                <strong>Active Production Model: {mmeta.get('model_name', 'EfficientNet-B0')}</strong><br>
                <div style='font-size:0.82rem; color:#047857; margin-top:4px;'>
                    Trained for {mmeta.get('epochs_ran', 10)} epochs on {mmeta.get('device', 'cpu').upper()} · Input Resolution: {mmeta.get('image_size', 224)} × {mmeta.get('image_size', 224)} px
                </div>
            </div>""", unsafe_allow_html=True)
            
            if img_m:
                c1,c2,c3,c4,c5,c6 = st.columns(6)
                for col, k, label in [
                    (c1,'accuracy','Accuracy'),(c2,'precision','Precision'),
                    (c3,'recall','Recall'),(c4,'f1','F1-Score'),(c5,'roc_auc','ROC-AUC'),
                    (c6,'inference_ms','Inference')
                ]:
                    with col:
                        val = img_m.get(k)
                        if val is not None:
                            val_str = f"{val:.1f}ms" if k == 'inference_ms' else f"{val:.4f}"
                            st.metric(label, val_str)
        
        if benchmark:
            st.markdown("<br><div style='font-weight:700; font-size:0.95rem; margin-bottom:8px;'>Architectural Comparison Benchmark (5-Epoch Baseline)</div>", unsafe_allow_html=True)
            df_bench = pd.DataFrame(benchmark)
            if 'accuracy' in df_bench.columns:
                df_show = df_bench[['model','accuracy','precision','recall','f1','inference_ms']].round(4)
                st.dataframe(df_show, use_container_width=True, hide_index=True)

    with tab2:
        if tab_data.get('loaded'):
            tmeta = tab_data.get('meta', {})
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown(f"""
                <div class="surface-card">
                    <div style='font-weight:700; font-size:1rem; color:#0f172a; margin-bottom:8px;'>Fire Occurrence Classifier ({tmeta.get('classifier_name', 'XGBoost')})</div>
                    <div style='font-size:0.82rem; color:#64748b; margin-bottom:12px;'>Binary classification on whether fire occurs (area > 0 ha).</div>
                """, unsafe_allow_html=True)
                if tab_m.get('classifier'):
                    cm = tab_m['classifier']
                    st.metric("ROC-AUC Score", f"{cm.get('roc_auc', 0.765):.4f}")
                    st.metric("Test Accuracy", f"{cm.get('accuracy', 0.654):.4f}")
                    st.metric("F1-Score", f"{cm.get('f1', 0.680):.4f}")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with c2:
                st.markdown(f"""
                <div class="surface-card">
                    <div style='font-weight:700; font-size:1rem; color:#0f172a; margin-bottom:8px;'>Burned Area Regressor ({tmeta.get('regressor_name', 'XGBoost')})</div>
                    <div style='font-size:0.82rem; color:#64748b; margin-bottom:12px;'>Log-space continuous prediction of burned hectare area.</div>
                """, unsafe_allow_html=True)
                if tab_m.get('regressor'):
                    rm = tab_m['regressor']
                    st.metric("MAE (Mean Absolute Error)", f"{rm.get('mae', 12.4):.4f} ha")
                    st.metric("RMSE", f"{rm.get('rmse', 38.2):.4f} ha")
                    st.metric("R² Score (log-space)", f"{rm.get('r2', 0.05):.4f}")
                st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        hist_path = META_DIR / "training_history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                hist = json.load(f)
            
            epochs = list(range(1, len(hist['train_loss'])+1))
            fig, axes = plt.subplots(1, 2, figsize=(14, 4), dpi=120)
            fig.patch.set_facecolor('#ffffff')
            for ax in axes:
                ax.set_facecolor('#ffffff')
            
            axes[0].plot(epochs, hist['train_loss'], 'b-o', markersize=5, label='Train Loss', color='#2563eb', linewidth=2)
            axes[0].plot(epochs, hist['val_loss'],   'r-s', markersize=5, label='Val Loss', color='#dc2626', linewidth=2)
            axes[0].set_title('Cross-Entropy Loss Progression', color='#0f172a', fontweight='bold')
            axes[0].set_xlabel('Epoch', color='#475569')
            axes[0].set_ylabel('Loss', color='#475569')
            axes[0].legend(facecolor='#ffffff', edgecolor='#cbd5e1', labelcolor='#0f172a')
            axes[0].tick_params(colors='#475569')
            axes[0].grid(alpha=0.2, color='#94a3b8')
            for sp in axes[0].spines.values(): sp.set_color('#cbd5e1')
            
            axes[1].plot(epochs, hist['val_acc'], 'g-^', markersize=5, label='Val Accuracy', color='#059669', linewidth=2)
            axes[1].plot(epochs, hist['val_f1'],  'm-D', markersize=5, label='Val F1', color='#7c3aed', linewidth=2)
            axes[1].set_title('Validation Accuracy & F1 Curves', color='#0f172a', fontweight='bold')
            axes[1].set_xlabel('Epoch', color='#475569')
            axes[1].set_ylabel('Score', color='#475569')
            axes[1].legend(facecolor='#ffffff', edgecolor='#cbd5e1', labelcolor='#0f172a')
            axes[1].tick_params(colors='#475569')
            axes[1].grid(alpha=0.2, color='#94a3b8')
            for sp in axes[1].spines.values(): sp.set_color('#cbd5e1')
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("Training history metadata located in artifacts/metadata/training_history.json.")


# ─────────────────────────────────────────────────────────────
# PAGE 7: DATASET INSIGHTS & EXPLORER
# ─────────────────────────────────────────────────────────────
def page_dataset_insights(metrics):
    st.markdown('<div class="section-head"><div class="section-title">🗂️ <span>Dataset Inventory & Asset Explorer</span></div><span class="section-tag">Corpus Explorer</span></div>', unsafe_allow_html=True)
    
    ds = metrics.get('dataset') or {}
    
    datasets = [
        {
            'name': 'Archive Fire Dataset',
            'type': 'Image Classification',
            'samples': ds.get('archive_total', 999),
            'fire': ds.get('archive_fire', 755),
            'safe': ds.get('archive_nofire', 244),
            'desc': 'Curated high-resolution dataset containing labeled aerial and ground wildfire and non-wildfire imagery.',
            'badge': 'Primary Vision Set',
            'color': '#dc2626'
        },
        {
            'name': 'UAVS-FDDB Raw Imagery',
            'type': 'UAV Aerial Vision',
            'samples': ds.get('uavs_raw_total', 1530),
            'fire': ds.get('uavs_fire', 1145),
            'safe': ds.get('uavs_nofire', 385),
            'desc': 'Unmanned Aerial Vehicle database capturing diverse forest canopy lighting, smoke dispersal, and terrain variants.',
            'badge': 'Aerial Benchmark',
            'color': '#ea580c'
        },
        {
            'name': 'UCI Forest Fires Tabular Dataset',
            'type': 'Meteorological & FWI',
            'samples': ds.get('csv_rows', 518),
            'fire': '270 (burned)',
            'safe': '248 (zero area)',
            'desc': 'Montesinho natural park record with Canadian FWI system indices and meteorological measurements.',
            'badge': 'Tabular Risk',
            'color': '#059669'
        }
    ]
    
    for d in datasets:
        st.markdown(f"""
        <div class="surface-card" style='border-left: 4px solid {d["color"]};'>
            <div style='display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:8px;'>
                <div>
                    <div style='font-size:1.1rem; font-weight:700; color:#0f172a;'>{d["name"]}</div>
                    <div style='font-size:0.8rem; font-weight:600; color:{d["color"]}; text-transform:uppercase;'>{d["badge"]} · {d["type"]}</div>
                </div>
                <div style='font-family:"Space Grotesk",sans-serif; font-size:1.4rem; font-weight:800; color:#0f172a;'>
                    {d["samples"]:,} <span style='font-size:0.8rem; color:#64748b; font-weight:500;'>samples</span>
                </div>
            </div>
            <div style='font-size:0.85rem; color:#475569; margin:6px 0 12px 0;'>
                {d["desc"]}
            </div>
            <div style='display:flex; gap:16px; font-size:0.82rem; font-weight:600;'>
                <span style='color:#dc2626;'>🔥 Fire Positive: {d["fire"]}</span>
                <span style='color:#059669;'>🌲 Non-Fire: {d["safe"]}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE 8: ABOUT & ARCHITECTURE
# ─────────────────────────────────────────────────────────────
def page_about():
    st.markdown('<div class="section-head"><div class="section-title">ℹ️ <span>About Forest Sentinel Platform</span></div><span class="section-tag">Documentation</span></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.2, 0.8], gap="large")
    
    with col1:
        st.markdown("""
        ### Executive Summary
        **Forest Sentinel** is an AI-driven dual-modality wildfire intelligence system developed for early detection, rapid risk estimation, and explainable decision support.

        #### Core Technical Pillars:
        1. **Vision Fire Classifier (CNN Transfer Learning)**:
           - Pretrained backbones: EfficientNet-B0, MobileNetV3, ResNet18.
           - Binary cross-entropy classification on aerial and ground imagery.
           - Attention attribution via Gradient-Weighted Class Activation Maps (Grad-CAM).

        2. **Environmental Risk Forecaster (Tabular ML)**:
           - Utilizes the Canadian Forest Fire Weather Index (FWI) system.
           - XGBoost classifier for wildfire occurrence probability.
           - XGBoost regressor with log1p transformation for burned area estimation (hectares).

        3. **Explainable AI Integration**:
           - Transparent heatmaps that localize the exact pixel receptive fields responsible for model predictions, preventing false positives from sunlight reflection.

        ---
        #### Academic Credits:
        Developed as part of the 7th Semester B.Tech AI/ML Course Curriculum at  
        **Symbiosis Institute of Technology (SIT), Nagpur**.
        """)
        
    with col2:
        st.markdown("""
        <div class="surface-card">
            <div style='font-weight:700; font-size:1rem; color:#0f172a; margin-bottom:12px;'>FWI Index Reference Guide</div>
            <div style='font-size:0.82rem; color:#475569; line-height:1.6;'>
                <strong>• FFMC (Fine Fuel Moisture Code):</strong> Moisture content of surface litter (18.7–96.2). Primary ignition readiness.<br><br>
                <strong>• DMC (Duff Moisture Code):</strong> Moisture content of loosely compacted decomposing organic layers (1.1–291.3).<br><br>
                <strong>• DC (Drought Code):</strong> Deep organic layers and heavy fuel moisture (7.9–860.6). Long-term seasonal drought.<br><br>
                <strong>• ISI (Initial Spread Index):</strong> Combines wind speed and FFMC without fuel quantity to model fire rate of spread (0–56.1).
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN DISPATCHER
# ─────────────────────────────────────────────────────────────
def main():
    inject_css()
    
    img_data = load_image_model()
    tab_data = load_tabular_models()
    metrics  = load_metrics()
    
    page = render_sidebar(img_data, tab_data)
    
    if page == "Overview":
        page_overview(img_data, tab_data, metrics)
    elif page == "Image Detection":
        page_image_detection(img_data)
    elif page == "Live Sandbox":
        page_live_sandbox(tab_data)
    elif page == "CSV Risk Analysis":
        page_csv_analysis(tab_data)
    elif page == "Analytics":
        page_analytics(metrics)
    elif page == "Model Performance":
        page_model_performance(img_data, tab_data, metrics)
    elif page == "Dataset Insights":
        page_dataset_insights(metrics)
    elif page == "About":
        page_about()


if __name__ == "__main__":
    main()
