"""
ARTVISION - Advertising & Marketing Suggestions Module
Connects fine art aesthetics with brand archetypes, color psychology,
commercial campaign themes, and consumer demographics.
"""

import streamlit as st
from src.database import get_all_styles, get_advertising_suggestions

STYLE_COLORS = {
    "Impressionism": "#4A7C59",
    "Cubism": "#B87333",
    "Abstract": "#2563EB",
    "Surrealism": "#7C3AED"
}


def render_advertising_page():
    st.markdown("""
        <div style="margin-bottom: 1.5rem;">
            <h2>📢 Advertising & Commercial Brand Intelligence</h2>
            <p style="color: #64748B; font-size: 1.05rem;">
                Translate artistic movements into high-impact commercial branding strategies. 
                Leverage art history to build compelling visual identities, brand archetypes, and campaign narratives.
            </p>
        </div>
    """, unsafe_allow_html=True)

    styles = get_all_styles()
    
    default_idx = 0
    if "current_style" in st.session_state and st.session_state["current_style"] in styles:
        default_idx = styles.index(st.session_state["current_style"])

    selected_style = st.selectbox(
        "Select Art Movement for Marketing Analysis:",
        options=styles,
        index=default_idx,
        key="ad_style_selector"
    )

    ad_info = get_advertising_suggestions(selected_style)
    if not ad_info:
        st.warning(f"No advertising strategy records found for {selected_style}.")
        return

    accent_color = STYLE_COLORS.get(selected_style, "#6B1D2F")

    # Brand Archetype Header Banner
    st.markdown(f"""
        <div class="art-card-highlight" style="border-left-color: {accent_color}; margin-top: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 0.85rem; text-transform: uppercase; color: {accent_color}; font-weight: 700; letter-spacing: 0.05em;">
                        Brand Identity & Archetype Alignment
                    </span>
                    <h2 style="margin: 0.25rem 0 0.5rem 0; color: #0F172A;">{selected_style} in Advertising</h2>
                </div>
                <div style="background: #EEF2F6; border-radius: 8px; padding: 6px 14px; font-weight: 700; color: #1E293B;">
                    Archetype: {ad_info['brand_archetype']}
                </div>
            </div>
            <p style="font-size: 1.05rem; color: #334155; line-height: 1.6; margin-top: 1rem;">
                {ad_info['marketing_rationale']}
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Core Strategy Dimensions (2x2 Grid)
    col_ind, col_emo = st.columns(2)

    with col_ind:
        st.markdown(f"""
            <div class="art-card">
                <h4 style="color: {accent_color}; margin-top: 0;">🏢 Prime Industry & Sector Fit</h4>
                <p style="color: #475569; font-size: 0.95rem; line-height: 1.6;">
                    {ad_info['industry_suitability']}
                </p>
                <h4 style="color: {accent_color}; margin-top: 1rem;">👥 Target Consumer Demographic</h4>
                <p style="color: #475569; font-size: 0.95rem; line-height: 1.6;">
                    {ad_info['target_demographic']}
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col_emo:
        st.markdown(f"""
            <div class="art-card">
                <h4 style="color: {accent_color}; margin-top: 0;">🧠 Emotional Impact & Consumer Psychology</h4>
                <p style="color: #475569; font-size: 0.95rem; line-height: 1.6;">
                    {ad_info['emotional_impact']}
                </p>
                <h4 style="color: {accent_color}; margin-top: 1rem;">🎨 Color Psychology & Visual Strategy</h4>
                <p style="color: #475569; font-size: 0.95rem; line-height: 1.6;">
                    {ad_info['color_psychology']}
                </p>
            </div>
        """, unsafe_allow_html=True)

    # Campaign Themes
    st.markdown("### 💡 Recommended Campaign Concepts & Slogans")
    themes = [t.strip().strip("'") for t in ad_info['campaign_themes'].split(",")]
    
    t_cols = st.columns(len(themes))
    for idx, theme in enumerate(themes):
        with t_cols[idx]:
            st.markdown(f"""
                <div class="art-card" style="text-align: center; border-top: 3px solid {accent_color};">
                    <span style="font-size: 0.75rem; text-transform: uppercase; color: #64748B;">Concept #{idx+1}</span>
                    <h4 style="margin: 6px 0; font-size: 1.1rem; color: #1E293B;">"{theme}"</h4>
                </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    render_advertising_page()
