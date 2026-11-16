"""
ARTVISION - About Project Module
Academic project presentation, system architecture, methodology,
evaluation metrics, and documentation for faculty evaluation.
"""

import os
import json
import streamlit as st
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


def render_about_page():
    st.markdown("""
        <div style="margin-bottom: 1.5rem;">
            <h2>📖 About the Academic Project</h2>
            <p style="color: #64748B; font-size: 1.05rem;">
                <b>Project Title:</b> ARTWORK STYLE CLASSIFICATION USING MACHINE LEARNING<br>
                <b>System Name:</b> ARTVISION Intelligent Curatorial & Visual Analytics Platform
            </p>
        </div>
    """, unsafe_allow_html=True)

    tab_overview, tab_arch, tab_metrics, tab_guide = st.tabs([
        "📋 Project Overview",
        "🧠 Neural Architecture & Tech Stack",
        "📊 Evaluation & Model Reports",
        "🛠️ Execution & Setup Manual"
    ])

    with tab_overview:
        st.markdown("""
            <div class="art-card">
                <h3 style="color: #6B1D2F; margin-top: 0;">Problem Statement & Motivation</h3>
                <p style="color: #334155; font-size: 0.95rem; line-height: 1.7;">
                    Art history curation has traditionally required decades of scholarly expertise to recognize 
                    nuanced brushwork, planar deconstruction, chromatic saturation, and compositional geometry. 
                    With the rapid digitization of cultural heritage (e.g. WikiArt, Google Arts & Culture, museum archives), 
                    automated classification using Computer Vision enables scalable cataloging, authentic art education, 
                    and innovative cross-disciplinary applications in spatial interior design and commercial advertising.
                </p>
                <h4 style="color: #6B1D2F; margin-top: 1rem;">Core Project Deliverables:</h4>
                <ul style="color: #475569; font-size: 0.95rem; line-height: 1.7;">
                    <li><b>Four-Class Deep Learning Classifier:</b> Distinguishing between Impressionism, Cubism, Abstract, and Surrealism.</li>
                    <li><b>Image Preprocessing Engine:</b> 224×224 Lanczos resampling, [-1, 1] tensor normalization, data augmentation.</li>
                    <li><b>Cultural Knowledge Base:</b> SQLite database containing comprehensive art history records and master artists.</li>
                    <li><b>Interior Color Clustering:</b> OpenCV K-Means spatial color extraction for room artwork pairing.</li>
                    <li><b>Pedagogical Curriculum:</b> Tiered learning roadmap for studio art students (Beginner, Intermediate, Advanced).</li>
                    <li><b>Commercial Strategy Engine:</b> Brand archetypes, color psychology, and advertising themes.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with tab_arch:
        st.markdown("""
            <div class="art-card">
                <h3 style="color: #6B1D2F; margin-top: 0;">Transfer Learning Pipeline</h3>
                <p style="color: #334155; font-size: 0.95rem; line-height: 1.7;">
                    The core classifier utilizes <b>MobileNetV2</b> pretrained on ImageNet (1.4 million images). 
                    MobileNetV2 employs inverted residual blocks with linear bottlenecks, providing high representational 
                    capacity with efficient compute footprint.
                </p>
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px; font-family: monospace; font-size: 0.85rem;">
                    Input Image (224×224×3)<br>
                    &nbsp;&nbsp;↓<br>
                    MobileNetV2 Feature Extractor (Frozen Weights)<br>
                    &nbsp;&nbsp;↓<br>
                    GlobalAveragePooling2D (7×7×1280 → 1280)<br>
                    &nbsp;&nbsp;↓<br>
                    BatchNormalization()<br>
                    &nbsp;&nbsp;↓<br>
                    Dense(256, activation='relu', L2 Regularization=1e-4)<br>
                    &nbsp;&nbsp;↓<br>
                    Dropout(0.4)<br>
                    &nbsp;&nbsp;↓<br>
                    Dense(128, activation='relu')<br>
                    &nbsp;&nbsp;↓<br>
                    Dropout(0.2)<br>
                    &nbsp;&nbsp;↓<br>
                    Dense(4, activation='softmax') → [Abstract, Cubism, Impressionism, Surrealism]
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 💻 Technology Stack Matrix")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("""
                <div class="art-card">
                    <b>Deep Learning & ML:</b><br>
                    • Python 3.10 - 3.14<br>
                    • TensorFlow / Keras (CNN & Transfer Learning)<br>
                    • Scikit-learn (Classification Report & Confusion Matrix)<br>
                    • NumPy & Pandas (Data processing)
                </div>
            """, unsafe_allow_html=True)
        with col_t2:
            st.markdown("""
                <div class="art-card">
                    <b>Computer Vision & UI:</b><br>
                    • OpenCV & Pillow (Resizing, Color Spaces, K-Means)<br>
                    • Streamlit (Interactive Modern Dashboard)<br>
                    • SQLite3 (Relational Knowledge Base)<br>
                    • Matplotlib & Seaborn (Analytical Plots)
                </div>
            """, unsafe_allow_html=True)

    with tab_metrics:
        st.markdown("### 📊 Empirical Model Performance Reports")
        
        acc_path = os.path.join(REPORTS_DIR, "accuracy.png")
        loss_path = os.path.join(REPORTS_DIR, "loss.png")
        cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
        metrics_json = os.path.join(REPORTS_DIR, "classification_report.json")

        if os.path.exists(acc_path) or os.path.exists(loss_path) or os.path.exists(cm_path):
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                if os.path.exists(acc_path):
                    st.image(acc_path, caption="Training vs Validation Accuracy Across Epochs", use_container_width=True)
                if os.path.exists(cm_path):
                    st.image(cm_path, caption="Test Set Confusion Matrix", use_container_width=True)
            with col_g2:
                if os.path.exists(loss_path):
                    st.image(loss_path, caption="Training vs Validation Loss", use_container_width=True)
                if os.path.exists(metrics_json):
                    with open(metrics_json, "r") as f:
                        data = json.load(f)
                    st.markdown("<b>Test Set Metrics Breakdown:</b>", unsafe_allow_html=True)
                    st.json(data)
        else:
            st.info(
                "📈 **No Training Artifacts Yet:** Once you run `python src/train_model.py`, the empirical "
                "training curves, validation loss graphs, and test set confusion matrix will automatically appear here."
            )

    with tab_guide:
        st.markdown("""
            <div class="art-card">
                <h3 style="color: #6B1D2F; margin-top: 0;">Step-by-Step Command Line Execution</h3>
                <p>Run these commands in your Windows PowerShell / Command Prompt:</p>
                <pre style="background: #1E293B; color: #F8FAFC; padding: 14px; border-radius: 8px;">
# 1. Navigate to Project Directory
cd C:\\Users\\vh566.LAPTOP-U33RR19L\\.gemini\\antigravity\\scratch\\artwork_style_classification

# 2. Install Project Dependencies
pip install -r requirements.txt

# 3. Initialize SQLite Database
python src/database.py

# 4. Generate Local Sample Verification Images
python src/sample_images.py

# 5. Populate Sample Dataset
python src/dataset_setup.py --populate-demo

# 6. Train Deep Learning Model
python src/train_model.py --epochs 12 --batch-size 16

# 7. Launch the Streamlit Interactive Dashboard
streamlit run app.py
                </pre>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    render_about_page()
