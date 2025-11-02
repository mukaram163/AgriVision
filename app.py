"""
🌿 AgriVision – Plant Disease Detection Dashboard
-------------------------------------------------
A professional Streamlit dashboard for real-time plant disease prediction.

Developed by: Mukaram Ali
GitHub: https://github.com/mukaram163/AgriVision
LinkedIn: https://linkedin.com/in/mukaram-ali-a05061279
"""

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.model import create_model, load_checkpoint

# ----------------------------------------------------------
# 🌱 Page Setup
# ----------------------------------------------------------
st.set_page_config(
    page_title="AgriVision 🌿",
    page_icon="🌱",
    layout="wide"
)

# ----------------------------------------------------------
# 🎨 Custom Styling (Green Theme + Accessibility Fixes)
# ----------------------------------------------------------
st.markdown("""
    <style>
    /* === GLOBAL BACKGROUND === */
    [data-testid="stAppViewContainer"] {
        background-color: #f8fff5;
        color: #1b4332;
    }
    [data-testid="stSidebar"] {
        background-color: #d8f3dc;
        color: #000 !important;
    }

    /* === HEADERS === */
    h1, h2, h3, h4, h5, h6 {
        color: #1b4332 !important;
        font-weight: 700 !important;
    }

    /* === File Uploader === */
    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] p,
    div[data-testid="stFileUploader"] small {
        color: #000 !important;
        font-weight: 700 !important;
    }
    div[data-testid="stFileUploaderDropzone"] {
        background-color: #ffffff !important;
        border: 2px dashed #1b4332 !important;
        border-radius: 10px !important;
    }
    div[data-testid="stFileUploaderDropzone"] * {
        color: #1b4332 !important;
    }
    div[data-testid="stFileUploaderBrowseButton"] {
        background-color: #1b4332 !important;
        color: white !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }

    /* === Info + Success Messages === */
    .stAlert, .stAlert p, .stAlert div {
        color: #000 !important;
        font-weight: 600 !important;
        background-color: #e9f5e9 !important;
        border-left: 5px solid #1b4332 !important;
    }
    .stSuccess, .stSuccess p, .stSuccess div, .stSuccess strong, .stSuccess span {
        background-color: #95d5b2 !important;
        color: #000 !important;
        border-radius: 10px;
        padding: 10px 14px !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
    }

    /* === Tabs + Typography === */
    button[data-baseweb="tab"], .stTabs [role="tab"] {
        color: #1b4332 !important;
        font-weight: 700 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 3px solid #1b4332 !important;
        color: #1b4332 !important;
    }

    /* === Image Captions === */
    .stImageCaption {
        color: #1b4332 !important;
        font-size: 0.9rem;
        font-weight: 600;
        background-color: rgba(255,255,255,0.7);
        padding: 4px 10px;
        border-radius: 6px;
        display: inline-block;
    }

    /* === General Text Fix (black text where needed) === */
    .stCaption, .stMarkdown p, .stInfo, .stInfo p {
        color: #000 !important;
        font-weight: 500 !important;
    }

    /* === Footer === */
    footer {visibility: hidden;}
    #footer-container {
        text-align: center;
        margin-top: 40px;
        color: #1b4332;
        font-weight: 500;
        font-size: 0.95rem;
    }
    #footer-container a {
        color: #1b4332;
        text-decoration: none;
        font-weight: 700;
    }
    #footer-container a:hover {
        color: #2d6a4f;
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# 🌿 Branding Header
# ----------------------------------------------------------
st.markdown("""
    <h1 style='text-align:center; color:#1b4332; font-weight:800;'>
        🌾 AgriVision – Plant Disease Detection Dashboard
    </h1>
    <p style='text-align:center; color:#2d6a4f; font-size:1.05rem;'>
        Empowering sustainable agriculture through AI-powered leaf disease detection 🌱
    </p>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# ⚙️ Sidebar Controls
# ----------------------------------------------------------
st.sidebar.title("🌿 AgriVision Controls")
st.sidebar.write("Upload a leaf image or view evaluation metrics.")
uploaded_file = st.sidebar.file_uploader("📤 Upload a leaf image...", type=["jpg", "jpeg", "png"])
show_metrics = st.sidebar.checkbox("Show Evaluation Metrics", value=True)

# ----------------------------------------------------------
# 🧠 Load Model (cached for performance)
# ----------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    model = create_model(num_classes=15)
    checkpoint = load_checkpoint("models/best_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

model = load_model()

class_names = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

# ----------------------------------------------------------
# 📊 Tabs
# ----------------------------------------------------------
tab1, tab2 = st.tabs(["🧠 Inference", "📈 Evaluation Metrics"])

# ----------------------------------------------------------
# 🧩 Inference Tab
# ----------------------------------------------------------
with tab1:
    st.header("🌿 Leaf Disease Prediction")
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

        # Image transformation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Model prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            predicted_class = class_names[pred_idx.item()]
            confidence = conf.item()

        with col2:
            st.subheader("🔍 Prediction Result")
            st.success(f"🌱 {predicted_class} ({confidence*100:.2f}% confidence)")

            # Confidence Bar Chart
            st.subheader("Confidence Breakdown (Top 5)")
            top_probs, top_idxs = torch.topk(probs[0], 5)
            plt.figure(figsize=(5, 3))
            plt.barh([class_names[i] for i in top_idxs], top_probs.cpu().numpy(), color="#1b4332")
            plt.gca().invert_yaxis()
            plt.xlabel("Confidence")
            plt.tight_layout()
            st.pyplot(plt.gcf(), clear_figure=True)

    else:
        st.info("📤 Upload a leaf image in the sidebar to get predictions.")

# ----------------------------------------------------------
# 📈 Metrics Tab
# ----------------------------------------------------------
with tab2:
    st.header("📊 Model Evaluation Metrics")

    if show_metrics and os.path.exists("results/eval_metrics.csv"):
        df = pd.read_csv("results/eval_metrics.csv")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("✅ Test Accuracy", f"{df['accuracy'].iloc[0]*100:.2f}%")
        with col2:
            st.metric("📉 Test Loss", f"{df['loss'].iloc[0]:.4f}")

        if "conf_matrix.csv" in os.listdir("results"):
            cm_df = pd.read_csv("results/conf_matrix.csv", index_col=0)
            st.subheader("📉 Confusion Matrix")
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_df, annot=True, fmt="d", cmap="Greens")
            st.pyplot(plt.gcf(), clear_figure=True)
        else:
            st.info("Confusion matrix not found. Run `python -m src.evaluate` to generate it.")
    else:
        st.warning("⚠️ Evaluation metrics not found. Please run `python -m src.evaluate` first.")

# ----------------------------------------------------------
# 💡 About Section
# ----------------------------------------------------------
st.markdown("""
---
### 💡 About This Project
**AgriVision** leverages deep learning (ResNet18 fine-tuned on the PlantVillage dataset) to identify plant diseases from leaf images.  
Trained on 50,000+ samples with 95%+ accuracy — built for agricultural researchers and smart-farming solutions.
""")

# ----------------------------------------------------------
# ❤️ Footer
# ----------------------------------------------------------
st.markdown("""
<div id="footer-container">
Developed with ❤️ by <b>Mukaram Ali</b> |
<a href="https://github.com/mukaram163/AgriVision" target="_blank">GitHub</a> · 
<a href="https://linkedin.com/in/mukaram-ali-a05061279" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)