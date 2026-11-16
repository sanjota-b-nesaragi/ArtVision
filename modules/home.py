"""
ARTVISION - Home / Dashboard View
Academic introduction, project objectives, style showcase, workflow, and system readiness dashboard.
"""

import os
import streamlit as st
from src.database import get_all_styles, get_style_details
from src.predict import is_model_available
from src.train_model import check_dataset_status


def render_home_page():
    # Hero Section
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">ARTVISION</div>
            <div style="font-size: 1.25rem; font-weight: 600; color: #D4AF37; margin-bottom: 0.5rem; letter-spacing: 0.05em;">
                ARTWORK STYLE CLASSIFICATION & CULTURAL INTELLIGENCE SYSTEM
            </div>
            <div class="hero-subtitle">
                An intelligent deep learning platform that classifies fine art into major historical movements
                using Transfer Learning (MobileNetV2 / EfficientNet) and synthesizes multi-dimensional insights across
                cultural heritage, interior design aesthetics, art education, and commercial advertising.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # System Status & Quick Action Row
    col_status1, col_status2, col_status3 = st.columns(3)

    model_ready = is_model_available()
    dataset_status = check_dataset_status()

    with col_status1:
        st.markdown("""
            <div class="metric-box">
                <div class="metric-label">Neural Network Engine</div>
                <div class="metric-value">MobileNetV2</div>
                <div style="margin-top: 6px;">
                    {}
                </div>
            </div>
        """.format(
            '<span class="status-badge-ready">● Model Ready</span>' if model_ready 
            else '<span class="status-badge-setup">⚠ Setup / Train Mode</span>'
        ), unsafe_allow_html=True)

    with col_status2:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Curated Dataset</div>
                <div class="metric-value">{dataset_status['total_images']} Images</div>
                <div style="margin-top: 6px;">
                    {'<span class="status-badge-ready">● Partitions Active</span>' if dataset_status['is_ready'] else '<span class="status-badge-setup">⚠ Setup Needed</span>'}
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col_status3:
        st.markdown("""
            <div class="metric-box">
                <div class="metric-label">Knowledge Base</div>
                <div class="metric-value">SQLite DB</div>
                <div style="margin-top: 6px;">
                    <span class="status-badge-ready">● 4 Styles Populated</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Academic Project Objectives
    st.markdown("### 🎯 Academic Project Objectives")
    col_obj1, col_obj2 = st.columns(2)

    with col_obj1:
        st.markdown("""
            <div class="art-card">
                <h4 style="color: #6B1D2F; margin-top: 0;">1. Deep Learning Vision Pipeline</h4>
                <p style="color: #475569; font-size: 0.95rem; line-height: 1.5;">
                    Implement an end-to-end Computer Vision system applying transfer learning with convolutional neural networks 
                    (MobileNetV2) to accurately categorize fine art paintings into four foundational movements: 
                    <b>Impressionism, Cubism, Abstract, and Surrealism</b>.
                </p>
                <h4 style="color: #6B1D2F; margin-top: 1rem;">2. Cultural & Heritage Preservation</h4>
                <p style="color: #475569; font-size: 0.95rem; line-height: 1.5;">
                    Preserve and disseminate historical knowledge by programmatically linking visual classifications with 
                    origin chronologies, technical methodologies, iconic masterworks, and influential art theorists.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col_obj2:
        st.markdown("""
            <div class="art-card">
                <h4 style="color: #6B1D2F; margin-top: 0;">3. Aesthetic Interior Recommendation</h4>
                <p style="color: #475569; font-size: 0.95rem; line-height: 1.5;">
                    Utilize OpenCV K-Means color clustering to analyze uploaded interior room photographs, extract dominant 
                    color palettes, and algorithmically recommend matching art styles based on chromatic harmony, lighting, and mood.
                </p>
                <h4 style="color: #6B1D2F; margin-top: 1rem;">4. Applied Guidance & Commercial Insights</h4>
                <p style="color: #475569; font-size: 0.95rem; line-height: 1.5;">
                    Provide tiered pedagogy (Beginner, Intermediate, Advanced) for practical fine art learning, along with 
                    brand archetypes and marketing strategy suggestions for commercial advertising campaigns.
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # The Four Art Movements Showcase
    st.markdown("### 🏛️ The Four Target Art Movements")
    styles = get_all_styles()
    style_cols = st.columns(4)

    style_highlights = {
        "Impressionism": {
            "period": "c. 1860s - 1890s",
            "pioneers": "Monet, Renoir, Degas",
            "accent": "#4A7C59",
            "tagline": "Light, atmosphere, and fleeting sensations."
        },
        "Cubism": {
            "period": "c. 1907 - 1920s",
            "pioneers": "Picasso, Braque, Gris",
            "accent": "#B87333",
            "tagline": "Multi-perspective geometric planar deconstruction."
        },
        "Abstract": {
            "period": "c. 1910 - 1960s",
            "pioneers": "Kandinsky, Mondrian, Pollock",
            "accent": "#2B5B84",
            "tagline": "Pure form, vibrant color fields, and emotion."
        },
        "Surrealism": {
            "period": "c. 1924 - 1950s",
            "pioneers": "Dalí, Magritte, Ernst",
            "accent": "#6D28D9",
            "tagline": "Subconscious dreamscapes and metaphysical juxtaposition."
        }
    }

    for idx, s_name in enumerate(styles):
        info = style_highlights.get(s_name, {})
        with style_cols[idx]:
            st.markdown(f"""
                <div class="art-card" style="border-top: 4px solid {info.get('accent', '#6B1D2F')}; min-height: 220px;">
                    <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748B; font-weight: 600;">{info.get('period', '')}</div>
                    <h3 style="margin: 0.35rem 0 0.5rem 0; font-size: 1.35rem;">{s_name}</h3>
                    <p style="font-size: 0.85rem; color: #475569; min-height: 48px;">{info.get('tagline', '')}</p>
                    <div style="font-size: 0.8rem; color: #0F172A; font-weight: 600; border-top: 1px solid #E2E8F0; padding-top: 8px;">
                        Key Artists: <br><span style="color: #64748B; font-weight: 400;">{info.get('pioneers', '')}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # How the System Works - Architecture Pipeline
    st.markdown("### ⚙️ How the System Operates")
    st.markdown("""
        <div class="art-card-highlight">
            <div style="display: flex; gap: 20px; align-items: center; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 220px;">
                    <b style="color: #6B1D2F;">Step 1: Input & Validation</b><br>
                    <span style="font-size: 0.85rem; color: #64748B;">Upload fine artwork (JPG/PNG/BMP) or room photos. Dimensions and file integrity validated.</span>
                </div>
                <div style="font-size: 1.5rem; color: #CBD5E1;">➔</div>
                <div style="flex: 1; min-width: 220px;">
                    <b style="color: #6B1D2F;">Step 2: Preprocessing</b><br>
                    <span style="font-size: 0.85rem; color: #64748B;">Bilinear Lanczos resize to 224×224, RGB conversion, and [-1, 1] normalization.</span>
                </div>
                <div style="font-size: 1.5rem; color: #CBD5E1;">➔</div>
                <div style="flex: 1; min-width: 220px;">
                    <b style="color: #6B1D2F;">Step 3: CNN Classification</b><br>
                    <span style="font-size: 0.85rem; color: #64748B;">Pretrained MobileNetV2 extracts deep visual embeddings with dense classification head.</span>
                </div>
                <div style="font-size: 1.5rem; color: #CBD5E1;">➔</div>
                <div style="flex: 1; min-width: 220px;">
                    <b style="color: #6B1D2F;">Step 4: Knowledge Synthesis</b><br>
                    <span style="font-size: 0.85rem; color: #64748B;">Retrieves cultural history, interior pairing, learning guidance, and advertising suggestions.</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Setup Status Notice Banner if Model not trained yet
    if not model_ready:
        st.info(
            "💡 **Setup State Notice:** The Deep Learning model file (`models/artwork_classifier.keras`) "
            "has not yet been compiled on this host. You can explore the sample datasets, test the Interior Design "
            "color analysis module, review cultural information, and run `python src/train_model.py` to train the full model."
        )


if __name__ == "__main__":
    render_home_page()
