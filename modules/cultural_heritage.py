"""
ARTVISION - Cultural & Heritage Information Module
Displays academic historical context, artistic techniques, cultural significance,
pioneering artists, and iconic masterpieces retrieved from the SQLite database.
"""

import streamlit as st
from src.database import get_all_styles, get_style_details

STYLE_COLORS = {
    "Impressionism": "#4A7C59",
    "Cubism": "#B87333",
    "Abstract": "#2563EB",
    "Surrealism": "#7C3AED"
}


def render_cultural_heritage_page():
    st.markdown("""
        <div style="margin-bottom: 1.5rem;">
            <h2>🏛️ Cultural & Historical Heritage</h2>
            <p style="color: #64748B; font-size: 1.05rem;">
                Explore the academic origins, sociopolitical impact, technical revolutions, and master artists 
                of foundational modern art movements stored in the ARTVISION SQLite knowledge base.
            </p>
        </div>
    """, unsafe_allow_html=True)

    styles = get_all_styles()
    
    # Default to current session prediction if available
    default_index = 0
    if "current_style" in st.session_state and st.session_state["current_style"] in styles:
        default_index = styles.index(st.session_state["current_style"])

    selected_style = st.selectbox(
        "Select Art Movement to Explore:",
        options=styles,
        index=default_index,
        help="Information is queried directly from the local SQLite database."
    )

    details = get_style_details(selected_style)

    if not details:
        st.error(f"No database records found for style '{selected_style}'.")
        return

    accent_color = STYLE_COLORS.get(selected_style, "#6B1D2F")

    # Header Card
    st.markdown(f"""
        <div class="art-card" style="border-left: 6px solid {accent_color}; margin-top: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 0.85rem; text-transform: uppercase; color: {accent_color}; font-weight: 700; letter-spacing: 0.05em;">
                        {details['origin_region']} &bull; {details['time_period']}
                    </span>
                    <h2 style="margin: 0.25rem 0 0.5rem 0; color: #0F172A;">{selected_style} Movement</h2>
                </div>
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 6px 14px; font-size: 0.85rem; font-weight: 600; color: #334155;">
                    ID: #{details['id']} &bull; Verified Record
                </div>
            </div>
            <p style="font-size: 1.05rem; color: #334155; line-height: 1.7; margin-top: 1rem;">
                {details['historical_background']}
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Core Academic Dimensions (2x2 Grid)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
            <div class="art-card">
                <h4 style="color: {accent_color}; margin-top: 0;">🎨 Distinctive Key Characteristics</h4>
                <p style="color: #475569; font-size: 0.95rem; line-height: 1.6;">
                    {details['key_characteristics']}
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="art-card">
                <h4 style="color: {accent_color}; margin-top: 0;">🖌️ Revolutionary Artistic Techniques</h4>
                <p style="color: #475569; font-size: 0.95rem; line-height: 1.6;">
                    {details['artistic_techniques']}
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="art-card">
                <h4 style="color: {accent_color}; margin-top: 0;">🌍 Sociocultural & Historical Significance</h4>
                <p style="color: #475569; font-size: 0.95rem; line-height: 1.6;">
                    {details['cultural_significance']}
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="art-card">
                <h4 style="color: {accent_color}; margin-top: 0;">🌈 Chromatic & Color Philosophy</h4>
                <p style="color: #475569; font-size: 0.95rem; line-height: 1.6;">
                    {details['color_philosophy']}
                </p>
            </div>
        """, unsafe_allow_html=True)

    # Prominent Artists & Masterpieces Section
    st.markdown("### 👑 Master Artists & Landmark Works")
    col_artists, col_works = st.columns(2)

    with col_artists:
        artists_list = [a.strip() for a in details['prominent_artists'].split(",")]
        st.markdown("""
            <div class="art-card">
                <h4 style="color: #0F172A; margin-top: 0;">Prominent Figures & Theorists</h4>
                <div style="margin-top: 10px;">
        """, unsafe_allow_html=True)
        for a in artists_list:
            st.markdown(f"<span class='style-pill'>✦ {a}</span>", unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    with col_works:
        st.markdown(f"""
            <div class="art-card">
                <h4 style="color: #0F172A; margin-top: 0;">Landmark Masterpieces</h4>
                <p style="color: #475569; font-size: 0.95rem; line-height: 1.6; margin-top: 10px;">
                    {details['iconic_masterpieces']}
                </p>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    render_cultural_heritage_page()
