"""
ARTVISION – Artwork Style Classification Using Machine Learning
================================================================
Academic Final Year Project Application
Integrates Deep Learning (MobileNetV2 / EfficientNetB0), OpenCV K-Means,
SQLite Knowledge Base, and Modern Streamlit Interface.
"""

import os
import sys

# Ensure project root is in Python sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

# Initialize SQLite database silently on startup
try:
    from src.database import init_database
    init_database()
except Exception as e:
    print(f"Warning initializing database: {e}")

# Page Configuration
st.set_page_config(
    page_title="ARTVISION – Artwork Style Classification",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Art-Inspired Custom CSS
CSS_PATH = os.path.join(PROJECT_ROOT, "assets", "css", "style.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Import Application Page Modules
from modules.home import render_home_page
from modules.classification import render_classification_page
from modules.cultural_heritage import render_cultural_heritage_page
from modules.interior_design import render_interior_design_page
from modules.skill_guidance import render_skill_guidance_page
from modules.advertising import render_advertising_page
from modules.about import render_about_page

# Status checkers
from src.predict import is_model_available
from src.train_model import check_dataset_status


def main():
    # Sidebar Branding
    st.sidebar.markdown("""
        <div style="text-align: center; padding: 10px 0 20px 0;">
            <div style="font-family: 'Cinzel', serif; font-size: 1.8rem; font-weight: 900; letter-spacing: 0.1em; color: #6B1D2F;">
                ARTVISION
            </div>
            <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748B;">
                Artwork Style Intelligence
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Navigation options
    PAGES = [
        "Home / Dashboard",
        "Artwork Classification",
        "Cultural & Heritage Information",
        "Interior Design Recommendations",
        "Art Learning Guidance",
        "Advertising Suggestions",
        "About Project"
    ]

    # Handle programmatic page switches via session state
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = PAGES[0]

    # Find index of current nav_page
    current_index = 0
    if st.session_state["nav_page"] in PAGES:
        current_index = PAGES.index(st.session_state["nav_page"])

    selected_page = st.sidebar.radio(
        "Navigation",
        options=PAGES,
        index=current_index,
        key="navigation_radio"
    )

    # Update session state if user clicks directly on radio
    st.session_state["nav_page"] = selected_page

    st.sidebar.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)

    # System Status Monitor in Sidebar
    model_ready = is_model_available()
    dataset_info = check_dataset_status()

    st.sidebar.markdown("### 🖥️ System Telemetry")
    if model_ready:
        st.sidebar.markdown("✅ **Model:** `MobileNetV2` (Loaded)")
    else:
        st.sidebar.markdown("⚠️ **Model:** `Setup Mode` (Not Trained)")

    st.sidebar.markdown(f"📁 **Dataset:** `{dataset_info['total_images']} Verified Images`")
    st.sidebar.markdown("🗄️ **Database:** `SQLite Active` (4 Styles)")

    st.sidebar.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)

    st.sidebar.markdown("""
        <div style="font-size: 0.75rem; color: #94A3B8; text-align: center; line-height: 1.5;">
            <b>Academic ML Project</b><br>
            Artwork Classification & Cultural AI<br>
            Python &bull; TensorFlow &bull; OpenCV &bull; Streamlit
        </div>
    """, unsafe_allow_html=True)

    # Main Page Router
    if selected_page == "Home / Dashboard":
        render_home_page()
    elif selected_page == "Artwork Classification":
        render_classification_page()
    elif selected_page == "Cultural & Heritage Information":
        render_cultural_heritage_page()
    elif selected_page == "Interior Design Recommendations":
        render_interior_design_page()
    elif selected_page == "Art Learning Guidance":
        render_skill_guidance_page()
    elif selected_page == "Advertising Suggestions":
        render_advertising_page()
    elif selected_page == "About Project":
        render_about_page()


if __name__ == "__main__":
    main()
