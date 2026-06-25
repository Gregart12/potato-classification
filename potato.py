import streamlit as st
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import joblib
import warnings
from skimage.feature import local_binary_pattern

# Completely silence background logs to maintain clean dashboard performance
warnings.filterwarnings('ignore')

# ==========================================
# 1. PROFESSIONAL SYSTEM THEMING & UI (CSS)
# ==========================================
st.set_page_config(
    page_title="AgriSync Tuber QA Engine",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Premium High-Contrast Dark Theme Workspace */
    .main {
        background: radial-gradient(circle at 50% 50%, #0d1117 0%, #06090e 100%);
        color: #e2e8f0;
        font-family: 'SF Pro Display', -apple-system, sans-serif;
    }
    .stApp {
        background: radial-gradient(circle at 50% 50%, #0d1117 0%, #06090e 100%);
    }
    
    /* Technical Dynamic Gradient Headers */
    h1 {
        background: linear-gradient(90deg, #58a6ff 0%, #4facfe 50%, #bc8cff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        letter-spacing: -1px;
        text-align: center;
    }
    
    /* Optimized Glassmorphism Dash Cards with Structural Borders */
    .dashboard-card {
        background: rgba(22, 27, 34, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(88, 166, 255, 0.2);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        margin-bottom: 25px;
        transition: all 0.3s ease;
    }
    .dashboard-card:hover {
        border: 1px solid rgba(188, 140, 255, 0.4);
        box-shadow: 0 8px 32px 0 rgba(188, 140, 255, 0.1);
    }
    
    /* Analytical Classification Badges */
    .badge-verdict {
        background: linear-gradient(135deg, #1f6feb, #8e44ad);
        color: #ffffff;
        padding: 6px 16px;
        border-radius: 12px;
        font-weight: bold;
        box-shadow: 0px 4px 15px rgba(31, 111, 235, 0.3);
    }
    
    /* Tuber Pathology Reference Box */
    .pathology-box {
        background: rgba(255, 255, 255, 0.03);
        border-left: 4px solid #58a6ff;
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

CATEGORIES = ["Black Scurf", "Blackleg", "Common Scab", "Dry Rot", "Healthy Potatoes", "Miscellaneous", "Pink Rot"]

# ==========================================
# 2. PATHOLOGICAL ENCYCLOPEDIA MATRIX
# ==========================================
DISEASE_DICTIONARY = {
    "Black Scurf": "Caused by the fungus Rhizoctonia solani. It forms dark brown or black hard structures (sclerotia) on the tuber surface. Often called 'the dirt that won't wash off', it ruins market value but doesn't cause deep structural rotting in early storage phases.",
    "Blackleg": "A severe bacterial infection caused by Pectobacterium species. It triggers a dark, slimy, wet rot that starts from the stem end and spreads inward. Highly infectious in wet conditions and causes rapid post-harvest breakdown.",
    "Common Scab": "A soil-borne bacterial defect caused by Streptomyces scabies. It results in rough, corky, raised or pitted brown lesions on the potato skin. It is purely cosmetic and skin-deep, meaning the interior tissue remains safe.",
    "Dry Rot": "One of the most destructive post-harvest fungal diseases, caused by Fusarium species. It causes the internal tuber tissue to shrink, collapse, and turn dark brown or black, often leaving hollow cavities filled with yellow or white fungal mold.",
    "Healthy Potatoes": "Premium quality specimen. Surface layers show minimal abrasions, balanced skin color profiles, intact cell walls, and complete absence of pathogenic structural microstructures or microbial decay indications.",
    "Miscellaneous": "Indicates general superficial anomalies including low-risk mechanical cuts, storage scuffs, sunburn greening, or surface dirt layers that do not show active infectious pathogen signatures.",
    "Pink Rot": "A rapid, water-borne fungal destruction caused by Phytophthora erythroseptica. The infected tissue remains rubbery but when cut open and exposed to air, the affected flesh undergoes a chemical reaction, turning bright pink within 15–30 minutes."
}

# ==========================================
# 3. CACHED INFRASTRUCTURE MODEL LOADERS
# ==========================================
@st.cache_resource
def load_all_classification_engines():
    engines = {
        "Random Forest": joblib.load('rf_model.pkl'),
        "XGBoost": joblib.load('xgb_model.pkl'),
        "LightGBM": joblib.load('lgb_model.pkl'),
        "SVM": joblib.load('svm_model.pkl'),
        "Logistic Regression": joblib.load('lr_model.pkl'),
        "Extra Trees": joblib.load('et_model.pkl'),
        "KNN": joblib.load('knn_model.pkl'),
        "Multi-Layer Perceptron": joblib.load('mlp_model.pkl')
    }
    custom_cnn = tf.keras.models.load_model('cnn_model.keras')
    mobilenet = tf.keras.models.load_model('mobilenet_model.keras')
    return engines, custom_cnn, mobilenet

try:
    ml_engines, cnn_m, mob_m = load_all_classification_engines()
except Exception as e:
    st.error(f"Engine Load Error: Verify pkl/keras files match workspace directory. Details: {e}")
    st.stop()

# ==========================================
# 4. COMPUTER VISION FEATURE ROUTINES
# ==========================================
def extract_ml_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 24, 3, 'uniform')
    n_bins = int(lbp.max() + 1)
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    hist_B = cv2.calcHist([img], [0], None, [16], [0, 256]).flatten()
    hist_G = cv2.calcHist([img], [1], None, [16], [0, 256]).flatten()
    hist_R = cv2.calcHist([img], [2], None, [16], [0, 256]).flatten()
    
    features = list(hist_B) + list(hist_G) + list(hist_R) + list(hist_lbp)
    return np.array(features).reshape(1, -1)

# ==========================================
# 5. CONTROL PANEL & ARCHITECTURE SELECTION
# ==========================================
st.sidebar.markdown("# ⚙️ System Processing Hub")
pipeline_selection = st.sidebar.radio(
    "Select AI Processing Cluster:",
    ["Single ML Model Mode", "2-3 Model Custom Ensemble", "Full 8-Model Consensus System", "Custom CNN Architecture", "MobileNetV2 (Transfer Learning)"]
)

if pipeline_selection == "Single ML Model Mode":
    target_single = st.sidebar.selectbox("Active Classifier Node:", list(ml_engines.keys()))
elif pipeline_selection == "2-3 Model Custom Ensemble":
    target_group = st.sidebar.multiselect("Select Cluster Sub-Models:", list(ml_engines.keys()), default=["Random Forest", "XGBoost"])

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Stream Specimen Asset (Drag & Drop):", type=["jpg", "jpeg", "png"])

# Dashboard Title
st.markdown("<h1>AGRISYNC // TUBER DIAGNOSTICS PROTOCOL</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e; font-size: 0.95rem; margin-bottom: 40px; letter-spacing:1px;'>EDGE-COMPUTING QUALITY ASSURANCE PIPELINE</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown("<div class='dashboard-card'><h3>📸 Scanned Specimen Profile</h3>", unsafe_allow_html=True)
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), use_container_width=True)
    else:
        st.info("Awaiting hardware scan validation input sequence.")
        st.image("https://images.unsplash.com/photo-1518977676601-b53f02bc6854?q=80&w=600", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='dashboard-card'><h3>📊 Diagnostic Engine Output Log</h3>", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        final_verdict_str = ""
        
        with st.spinner("Decoding microstructural layers..."):
            
            # --- EVAL BLOCK 1: SINGLE MODEL ---
            if pipeline_selection == "Single ML Model Mode":
                vec = extract_ml_features(cv2.resize(raw_img, (128, 128)))
                model = ml_engines[target_single]
                pred = model.predict(vec)[0]
                final_verdict_str = CATEGORIES[pred]
                
                st.markdown(f"<h4>System Output: <span class='badge-verdict'>{final_verdict_str}</span></h4>", unsafe_allow_html=True)
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(vec)[0]
                    st.metric(label="Target Probability Edge", value=f"{probs[pred]*100:.2f}%")

            # --- EVAL BLOCK 2: CUSTOM ENSEMBLE ---
            elif pipeline_selection == "2-3 Model Custom Ensemble" and (2 <= len(target_group) <= 3):
                vec = extract_ml_features(cv2.resize(raw_img, (128, 128)))
                votes = [ml_engines[m_name].predict(vec)[0] for m_name in target_group]
                consensus = max(set(votes), key=votes.count)
                final_verdict_str = CATEGORIES[consensus]
                agreement = (votes.count(consensus) / len(votes)) * 100
                
                st.markdown(f"<h4>Consensus Verdict: <span class='badge-verdict'>{final_verdict_str}</span></h4>", unsafe_allow_html=True)
                st.metric(label="Ensemble Consensus Agreement Rate", value=f"{agreement:.1f}%")

            # --- EVAL BLOCK 3: ALL 8 APPROACHES ---
            elif pipeline_selection == "Full 8-Model Consensus System":
                vec = extract_ml_features(cv2.resize(raw_img, (128, 128)))
                all_votes = []
                detailed_results = {}
                
                for name, model in ml_engines.items():
                    pred_class_idx = model.predict(vec)[0]
                    all_votes.append(pred_class_idx)
                    probs = model.predict_proba(vec)[0] if hasattr(model, "predict_proba") else None
                    detailed_results[name] = {
                        "Verdict": CATEGORIES[pred_class_idx],
                        "Confidence": f"{probs[pred_class_idx]*100:.1f}%" if probs is not None else "N/A"
                    }
                
                consensus = max(set(all_votes), key=all_votes.count)
                final_verdict_str = CATEGORIES[consensus]
                consensus_rate = (all_votes.count(consensus) / 8) * 100
                
                st.markdown(f"<h4>Consensus Decision: <span class='badge-verdict'>{final_verdict_str}</span></h4>", unsafe_allow_html=True)
                st.metric(label="Network Voting Consensus Rate", value=f"{consensus_rate:.1f}%")
                
                df_results = pd.DataFrame.from_dict(detailed_results, orient='index')
                st.dataframe(df_results, use_container_width=True)

            # --- EVAL BLOCK 4: CUSTOM CNN ---
            elif pipeline_selection == "Custom CNN Architecture":
                img_proc = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                img_proc = cv2.resize(img_proc, (128, 128)) / 255.0
                res_probs = cnn_m.predict(np.expand_dims(img_proc, axis=0))[0]
                pred = np.argmax(res_probs)
                final_verdict_str = CATEGORIES[pred]
                
                st.markdown(f"<h4>CNN Class Resolution: <span class='badge-verdict'>{final_verdict_str}</span></h4>", unsafe_allow_html=True)
                st.bar_chart(dict(zip(CATEGORIES, res_probs.tolist())))

            # --- EVAL BLOCK 5: CHAMPION MOBILENETV2 ---
            elif pipeline_selection == "MobileNetV2 (Transfer Learning)":
                img_proc = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                img_proc = cv2.resize(img_proc, (160, 160))
                img_proc = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(img_proc, axis=0))
                res_probs = mob_m.predict(img_proc)[0]
                pred = np.argmax(res_probs)
                final_verdict_str = CATEGORIES[pred]
                
                st.markdown(f"<h4>MobileNetV2 Network Verdict: <span class='badge-verdict'>{final_verdict_str}</span></h4>", unsafe_allow_html=True)
                st.metric(label="Transfer Learning Feature Proximity", value=f"{res_probs[pred]*100:.2f}%")
                st.bar_chart(dict(zip(CATEGORIES, res_probs.tolist())))

        # --- INFRASTRUCTURE PATHOLOGY COGNITIVE DISPLAY ---
        if final_verdict_str in DISEASE_DICTIONARY:
            st.markdown("<div class='pathology-box'>", unsafe_allow_html=True)
            st.markdown(f"🔬 **Pathology Insights: {final_verdict_str}**")
            st.markdown(f"<p style='font-size:0.9rem; color:#a0aec0; margin-top:5px;'>{DISEASE_DICTIONARY[final_verdict_str]}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.write("Awaiting target asset drop sequence to trigger smart analytics logs.")
        
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr style='border-top: 1px solid #21262d; margin-top: 60px;'><p style='text-align: center; color: #8b949e; font-size: 11px;'>AgriSync Quality Evaluation Node • Deployed under low-memory local constraints.</p>", unsafe_allow_html=True)