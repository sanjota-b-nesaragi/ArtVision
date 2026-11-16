"""
ARTVISION - Artwork Style Classification Module
Handles image input, image validation, neural network inference,
probability distribution visualization, and cross-module state linking.
"""

import os
import io
import streamlit as st
from PIL import Image

from src.predict import (
    predict_artwork_style,
    mock_predict_for_setup_preview,
    is_model_available,
    ModelNotTrainedError
)
from src.preprocessing import validate_image_file, ImageValidationError

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DATA_DIR = os.path.join(PROJECT_ROOT, "dataset", "sample_data")
CLASS_COLORS = {
    "Impressionism": "#4A7C59",
    "Cubism": "#B87333",
    "Abstract": "#2563EB",
    "Surrealism": "#7C3AED"
}


def render_classification_page():
    st.markdown("""
        <div style="margin-bottom: 1.5rem;">
            <h2>🎨 Artwork Style Classification</h2>
            <p style="color: #64748B; font-size: 1.05rem;">
                Upload a painting image to classify its movement into <b>Impressionism, Cubism, Abstract, or Surrealism</b>
                using deep convolutional transfer learning.
            </p>
        </div>
    """, unsafe_allow_html=True)

    model_ready = is_model_available()

    if not model_ready:
        st.warning(
            "⚠️ **Setup Mode Notice:** A trained model file (`models/artwork_classifier.keras`) is not yet present on disk. "
            "You can preview image validation, UI preprocessing, and sample testing below. To train the actual model, execute: `python src/train_model.py`."
        )

    # Input Method Selection
    input_tab1, input_tab2 = st.tabs(["📤 Upload Painting File", "🖼️ Test with Curated Samples"])

    selected_image = None
    image_source_name = "Uploaded Image"

    with input_tab1:
        uploaded_file = st.file_uploader(
            "Select an artwork painting image (JPG, JPEG, PNG, WEBP, BMP):",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            help="High-resolution painting photos or scans produce optimal visual feature extraction."
        )
        if uploaded_file is not None:
            selected_image = uploaded_file
            image_source_name = uploaded_file.name

    with input_tab2:
        st.markdown("<p style='font-size: 0.9rem; color: #475569;'>Choose a verified academic sample piece from the dataset to test immediately:</p>", unsafe_allow_html=True)
        
        sample_options = {
            "Impressionism (Sunlit Water Lilies & Garden)": "sample_impressionism.bmp",
            "Cubism (Planar Geometric Facets & Ochres)": "sample_cubism.bmp",
            "Abstract (Neoplastic Color Fields & Dynamic Grid)": "sample_abstract.bmp",
            "Surrealism (Metaphysical Dreamscape & Celestial Sphere)": "sample_surrealism.bmp"
        }

        sample_cols = st.columns(4)
        for idx, (label, fname) in enumerate(sample_options.items()):
            fpath = os.path.join(SAMPLE_DATA_DIR, fname)
            with sample_cols[idx]:
                if os.path.exists(fpath):
                    try:
                        thumb = Image.open(fpath)
                        st.image(thumb, use_container_width=True)
                    except Exception:
                        pass
                    if st.button(f"Select {fname.split('_')[1].split('.')[0].title()}", key=f"btn_{fname}"):
                        st.session_state["selected_sample_path"] = fpath
                        st.session_state["selected_sample_name"] = fname

        if "selected_sample_path" in st.session_state and os.path.exists(st.session_state["selected_sample_path"]):
            selected_image = st.session_state["selected_sample_path"]
            image_source_name = st.session_state.get("selected_sample_name", "Sample Artwork")
            st.success(f"Selected: **{image_source_name}**")

    # If an image has been selected or uploaded
    if selected_image is not None:
        st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)

        col_preview, col_results = st.columns([1, 1.25])

        with col_preview:
            st.markdown("#### 🖼️ Image Preview & Inspection")
            try:
                if isinstance(selected_image, str):
                    pil_display = Image.open(selected_image)
                    file_size_kb = round(os.path.getsize(selected_image) / 1024, 1)
                else:
                    selected_image.seek(0)
                    pil_display = Image.open(selected_image)
                    file_size_kb = round(len(selected_image.getvalue()) / 1024, 1)

                st.image(pil_display, caption=f"{image_source_name} ({pil_display.width}×{pil_display.height} px, {file_size_kb} KB)", use_container_width=True)

                # Preprocessing Verification Badge
                is_valid, msg = validate_image_file(selected_image)
                if is_valid:
                    st.markdown("""
                        <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 10px 14px; margin-top: 10px;">
                            <span style="color: #166534; font-weight: 600; font-size: 0.85rem;">✓ Preprocessing Validation Passed</span><br>
                            <span style="color: #15803D; font-size: 0.8rem;">Ready for 224×224 Lanczos resampling and [-1, 1] normalization.</span>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"Validation Warning: {msg}")

            except Exception as e:
                st.error(f"Error reading image: {e}")
                return

            analyze_btn = st.button("🚀 Classify Artwork Style", type="primary", use_container_width=True)

        with col_results:
            st.markdown("#### 📊 Classification Intelligence")

            # Trigger prediction on button click or if already cached
            if analyze_btn:
                with st.spinner("Executing neural feature extraction across 4 art movements..."):
                    try:
                        if model_ready:
                            result = predict_artwork_style(selected_image, filename=image_source_name)
                        else:
                            # Clearly labeled setup preview mode
                            result = mock_predict_for_setup_preview(selected_image, filename=image_source_name)

                        st.session_state["current_prediction"] = result
                        st.session_state["current_style"] = result["predicted_style"]
                        st.session_state["analyzed_image"] = pil_display

                    except ModelNotTrainedError as mnfe:
                        st.error(f"Model Error: {mnfe}")
                        return
                    except ImageValidationError as ive:
                        st.error(f"Image Error: {ive}")
                        return
                    except Exception as ex:
                        st.error(f"Inference Exception: {ex}")
                        return

            if "current_prediction" in st.session_state:
                res = st.session_state["current_prediction"]
                style = res["predicted_style"]
                conf = res["confidence"]
                probs = res["probabilities"]
                is_real = res["is_trained_model"]

                accent_color = CLASS_COLORS.get(style, "#6B1D2F")

                # Result Banner Card
                st.markdown(f"""
                    <div class="prediction-badge" style="border-top: 4px solid {accent_color};">
                        <div class="prediction-title">Primary Identified Art Style</div>
                        <div class="prediction-style-name">{style}</div>
                        <div class="prediction-confidence">
                            Model Confidence: <b>{conf:.1f}%</b>
                        </div>
                        {'' if is_real else '<div style="font-size: 0.8rem; color: #FBBF24; margin-top: 6px;">[Setup Demonstration Mode - Train model for actual weights]</div>'}
                    </div>
                """, unsafe_allow_html=True)

                # Probabilities Distribution Visualization
                st.markdown("##### 📈 Softmax Probability Distribution:")
                for class_name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                    c_color = CLASS_COLORS.get(class_name, "#6B1D2F")
                    st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 2px;">
                            <span><b>{class_name}</b></span>
                            <span style="font-weight: 600; color: {c_color};">{prob:.1f}%</span>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(float(min(1.0, max(0.0, prob / 100.0))))

                st.markdown("<br>", unsafe_allow_html=True)

                # Cross-Module Exploration Shortcuts
                st.markdown("##### 🧭 Explore In-Depth Modules for this Style:")
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("🏛️ Cultural & Historical Info", use_container_width=True):
                        st.session_state["nav_page"] = "Cultural & Heritage Information"
                        st.rerun()
                    if st.button("🎨 Art Learning Guidance", use_container_width=True):
                        st.session_state["nav_page"] = "Art Learning Guidance"
                        st.rerun()
                with col_btn2:
                    if st.button("🏡 Interior Recommendations", use_container_width=True):
                        st.session_state["nav_page"] = "Interior Design Recommendations"
                        st.rerun()
                    if st.button("📢 Advertising Suggestions", use_container_width=True):
                        st.session_state["nav_page"] = "Advertising Suggestions"
                        st.rerun()
    else:
        st.info("👆 Please upload a painting file or select one of the curated test samples above to begin analysis.")


if __name__ == "__main__":
    render_classification_page()
