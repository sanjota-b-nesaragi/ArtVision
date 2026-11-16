"""
ARTVISION - Interior Design Recommendation Module
Analyzes room interiors via OpenCV K-Means color clustering, computes chromatic warmth,
luminosity, and saturation, and pairs living spaces with optimal artwork styles and framing.
"""

import os
import streamlit as st
from PIL import Image

from src.interior_recommendation import (
    extract_dominant_colors,
    recommend_artwork_for_room
)
from src.database import get_interior_recommendations

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DATA_DIR = os.path.join(PROJECT_ROOT, "dataset", "sample_data")


def render_interior_design_page():
    st.markdown("""
        <div style="margin-bottom: 1.5rem;">
            <h2>🏡 Interior Design Recommendation Engine</h2>
            <p style="color: #64748B; font-size: 1.05rem;">
                Upload a photograph of your living space or room interior. ARTVISION analyzes its chromatic composition,
                lighting temperature, and architectural mood using <b>K-Means Color Clustering</b> to recommend
                the most harmonious artwork styles, framing profiles, and hanging placement.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.info(
        "💡 **Note on Room Analysis:** Room interiors are processed through an intelligent OpenCV color clustering "
        "and architectural balance engine, rather than the artwork painting classifier. This ensures recommendations "
        "are tailored strictly to spatial aesthetics and chromatic equilibrium."
    )

    tab_upload, tab_sample = st.tabs(["📤 Upload Room Photo", "🛋️ Choose Curated Room Interiors"])

    selected_room = None
    room_name = "Uploaded Room"

    with tab_upload:
        uploaded_room = st.file_uploader(
            "Upload an interior photo (JPG, PNG, WEBP, BMP):",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="room_uploader"
        )
        if uploaded_room is not None:
            selected_room = uploaded_room
            room_name = uploaded_room.name

    with tab_sample:
        st.markdown("<p style='font-size: 0.9rem; color: #475569;'>Test with calibrated interior architectural spaces:</p>", unsafe_allow_html=True)
        room_samples = {
            "Warm Sunlit Living Room (Oak & Terracotta)": "room_warm_living.bmp",
            "Cool Minimalist Loft (Slate & Concrete)": "room_minimalist_loft.bmp",
            "Moody Library / Den (Dark Navy & Walnut)": "room_moody_den.bmp"
        }
        r_cols = st.columns(3)
        for idx, (label, fname) in enumerate(room_samples.items()):
            fpath = os.path.join(SAMPLE_DATA_DIR, fname)
            with r_cols[idx]:
                if os.path.exists(fpath):
                    try:
                        thumb = Image.open(fpath)
                        st.image(thumb, caption=label, use_container_width=True)
                    except Exception:
                        pass
                    if st.button(f"Select {fname.split('_')[1].title()} Room", key=f"r_btn_{fname}"):
                        st.session_state["selected_room_path"] = fpath
                        st.session_state["selected_room_label"] = label

        if "selected_room_path" in st.session_state and os.path.exists(st.session_state["selected_room_path"]):
            selected_room = st.session_state["selected_room_path"]
            room_name = st.session_state.get("selected_room_label", "Curated Room")
            st.success(f"Selected: **{room_name}**")

    if selected_room is not None:
        st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)

        col_img, col_analysis = st.columns([1, 1.3])

        with col_img:
            st.markdown("#### 🖼️ Room Visual Inspection")
            try:
                if isinstance(selected_room, str):
                    room_pil = Image.open(selected_room)
                else:
                    selected_room.seek(0)
                    room_pil = Image.open(selected_room)

                st.image(room_pil, caption=f"{room_name} ({room_pil.width}×{room_pil.height} px)", use_container_width=True)
            except Exception as e:
                st.error(f"Failed to display image: {e}")
                return

            analyze_room_btn = st.button("✨ Analyze Room Color & Harmony", type="primary", use_container_width=True)

        with col_analysis:
            st.markdown("#### 🎨 Dominant Palette & Spatial Metrics")

            if analyze_room_btn or "room_palette" in st.session_state:
                with st.spinner("Extracting k-means color clusters and calculating temperature..."):
                    try:
                        palette = extract_dominant_colors(selected_room, k=5)
                        recommendation = recommend_artwork_for_room(palette)
                        st.session_state["room_palette"] = palette
                        st.session_state["room_recommendation"] = recommendation
                    except Exception as err:
                        st.error(f"Analysis error: {err}")
                        return

                palette = st.session_state["room_palette"]
                rec = st.session_state["room_recommendation"]

                # Render Swatches
                swatch_html = "<div class='swatch-container'>"
                for c in palette:
                    swatch_html += f"""
                        <div class="swatch-box">
                            <div class="swatch-color" style="background-color: {c['hex']};"></div>
                            <span class="swatch-hex">{c['hex']}</span>
                            <span class="swatch-pct">{c['percentage']}% ({c['temperature']})</span>
                        </div>
                    """
                swatch_html += "</div>"
                st.markdown(swatch_html, unsafe_allow_html=True)

                # Room Metrics Summary
                m = rec["room_metrics"]
                st.markdown(f"""
                    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px; margin-top: 10px;">
                        <span style="font-size: 0.85rem; color: #64748B;">Spatial Chromatic Distribution:</span><br>
                        <b>{m['warm_percentage']}% Warm</b> &bull; 
                        <b>{m['cool_percentage']}% Cool</b> &bull; 
                        <b>{m['neutral_percentage']}% Neutral</b> &bull; 
                        <span>Brightness Index: <b>{m['average_brightness']}</b></span>
                    </div>
                """, unsafe_allow_html=True)

        # Style Recommendation Cards
        if "room_recommendation" in st.session_state:
            rec = st.session_state["room_recommendation"]
            pri_style = rec["primary_style"]
            sec_style = rec["secondary_style"]
            pri_info = rec["primary_details"]

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🏆 Algorithmic Art Style Recommendations")

            st.markdown(f"""
                <div class="art-card-highlight">
                    <span style="font-size: 0.85rem; text-transform: uppercase; color: #D4AF37; font-weight: 700; letter-spacing: 0.05em;">
                        Primary Curatorial Recommendation
                    </span>
                    <h2 style="margin: 0.25rem 0 0.5rem 0; color: #0F172A;">{pri_style}</h2>
                    <p style="font-size: 1.05rem; color: #334155; line-height: 1.6;">
                        {rec['rationale']}
                    </p>
                    <div style="font-size: 0.9rem; color: #64748B; margin-top: 10px;">
                        <b>Secondary Complementary Style:</b> {sec_style}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            if pri_info:
                st.markdown("#### 📐 Curatorial Framing & Placement Specifications")
                col_spec1, col_spec2 = st.columns(2)

                with col_spec1:
                    st.markdown(f"""
                        <div class="art-card">
                            <h4 style="color: #6B1D2F; margin-top: 0;">🖼️ Recommended Framing</h4>
                            <p style="color: #475569; font-size: 0.95rem; line-height: 1.6;">
                                {pri_info['framing_advice']}
                            </p>
                            <h4 style="color: #6B1D2F; margin-top: 1rem;">💡 Optimal Lighting</h4>
                            <p style="color: #475569; font-size: 0.95rem; line-height: 1.6;">
                                {pri_info['lighting_requirements']}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)

                with col_spec2:
                    st.markdown(f"""
                        <div class="art-card">
                            <h4 style="color: #6B1D2F; margin-top: 0;">📍 Focal Placement Tips</h4>
                            <p style="color: #475569; font-size: 0.95rem; line-height: 1.6;">
                                {pri_info['placement_tips']}
                            </p>
                            <h4 style="color: #6B1D2F; margin-top: 1rem;">🏛️ Compatible Architectural Aesthetics</h4>
                            <p style="color: #475569; font-size: 0.95rem; line-height: 1.6;">
                                {pri_info['architectural_styles']}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)


if __name__ == "__main__":
    render_interior_design_page()
