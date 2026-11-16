"""
ARTVISION - Skill-Based Art Learning Guidance Module
Provides structured pedagogy for mastering painting techniques across three tiers:
Beginner, Intermediate, and Advanced, queried from the SQLite database.
"""

import streamlit as st
from src.database import get_all_styles, get_skill_guidance

STYLE_COLORS = {
    "Impressionism": "#4A7C59",
    "Cubism": "#B87333",
    "Abstract": "#2563EB",
    "Surrealism": "#7C3AED"
}


def render_skill_guidance_page():
    st.markdown("""
        <div style="margin-bottom: 1.5rem;">
            <h2>🎨 Skill-Based Art Learning Curriculum</h2>
            <p style="color: #64748B; font-size: 1.05rem;">
                Develop technical mastery in your chosen art movement. Follow structured, studio-tested curriculum 
                spanning materials, core mechanics, step-by-step studio drills, and master copy studies.
            </p>
        </div>
    """, unsafe_allow_html=True)

    styles = get_all_styles()
    
    default_idx = 0
    if "current_style" in st.session_state and st.session_state["current_style"] in styles:
        default_idx = styles.index(st.session_state["current_style"])

    selected_style = st.selectbox(
        "Select Art Movement for Practical Studio Learning:",
        options=styles,
        index=default_idx,
        key="skill_style_selector"
    )

    accent_color = STYLE_COLORS.get(selected_style, "#6B1D2F")

    # Fetch guides from DB
    guides = get_skill_guidance(selected_style)
    if not guides:
        st.warning(f"No curriculum records found for {selected_style}.")
        return

    # Map guides by difficulty
    tier_map = {g["difficulty_level"].lower(): g for g in guides}

    # Difficulty Tabs
    tab_beg, tab_int, tab_adv = st.tabs([
        "🌱 Beginner Level (Fundamentals)",
        "🌿 Intermediate Level (Technique & Space)",
        "🌳 Advanced Level (Mastery & Vision)"
    ])

    for tab, level_key, badge_label in [
        (tab_beg, "beginner", "Foundations & Mark-Making"),
        (tab_int, "intermediate", "Compositional Depth & Optical Mastery"),
        (tab_adv, "advanced", "Monumental Synthesis & Voice")
    ]:
        with tab:
            guide = tier_map.get(level_key)
            if not guide:
                st.info("Curriculum details pending for this tier.")
                continue

            st.markdown(f"""
                <div class="art-card" style="border-left: 5px solid {accent_color}; margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="margin: 0; color: #0F172A;">{selected_style} &bull; {guide['difficulty_level']} Tier</h3>
                        <span class="style-pill" style="background: #FEF3C7; color: #92400E; font-weight: 700;">{badge_label}</span>
                    </div>
                    <div style="margin-top: 12px;">
                        <b style="color: #6B1D2F;">🎯 Learning Objectives:</b>
                        <p style="color: #334155; font-size: 1rem; line-height: 1.6; margin-top: 4px;">
                            {guide['learning_objectives']}
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            col_mat, col_tech = st.columns(2)

            with col_mat:
                st.markdown(f"""
                    <div class="art-card">
                        <h4 style="color: #6B1D2F; margin-top: 0;">📦 Required Materials & Tools</h4>
                        <p style="color: #475569; font-size: 0.95rem; line-height: 1.6;">
                            {guide['required_materials']}
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            with col_tech:
                st.markdown(f"""
                    <div class="art-card">
                        <h4 style="color: #6B1D2F; margin-top: 0;">🖌️ Core Technical Mechanics</h4>
                        <p style="color: #475569; font-size: 0.95rem; line-height: 1.6;">
                            {guide['core_techniques']}
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            # Step-by-Step Exercises
            exercises = guide['step_by_step_exercises'].split("\n")
            st.markdown("#### 📝 Studio Practice Exercises")
            for ex in exercises:
                if ex.strip():
                    st.markdown(f"""
                        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;">
                            <span style="color: #1E293B; font-size: 0.95rem;">{ex}</span>
                        </div>
                    """, unsafe_allow_html=True)

            col_imp, col_pit = st.columns(2)

            with col_imp:
                st.markdown(f"""
                    <div class="art-card">
                        <h4 style="color: #059669; margin-top: 0;">🚀 Daily Improvement Activities</h4>
                        <p style="color: #475569; font-size: 0.95rem; line-height: 1.6;">
                            {guide['improvement_activities']}
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            with col_pit:
                st.markdown(f"""
                    <div class="art-card">
                        <h4 style="color: #DC2626; margin-top: 0;">⚠️ Common Pitfalls to Avoid</h4>
                        <p style="color: #475569; font-size: 0.95rem; line-height: 1.6;">
                            {guide['common_pitfalls']}
                        </p>
                    </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    render_skill_guidance_page()
