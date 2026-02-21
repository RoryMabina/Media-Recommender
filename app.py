"""
Media Recommender — Streamlit UI
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
from recommender import MediaRecommender, load_data

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="What Should I Watch?",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 What Should I Watch?")
st.write("Tell us what you've watched and we'll find your next obsession — movies, shows, and anime.")

# ─────────────────────────────────────────────
# Load model (cached so it only runs once)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading recommendation engine…")
def get_recommender():
    df = load_data()
    rec = MediaRecommender()
    rec.fit(df)
    return rec

try:
    rec = get_recommender()
    dataset_loaded = True
except FileNotFoundError as e:
    st.error(str(e))
    st.info("📥 Download the datasets by following the instructions in the README, then place them in the `data/` folder.")
    dataset_loaded = False

# ─────────────────────────────────────────────
# Sidebar — settings
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    n_recs = st.slider("Number of recommendations", 5, 25, 10)
    filter_type = st.selectbox(
        "Filter by type",
        ["All", "Movie", "TV Show", "Anime"],
    )
    filter_type = None if filter_type == "All" else filter_type

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown(
        "This recommender uses **content-based filtering** — it analyses genres and "
        "descriptions to find titles similar to what you've enjoyed.\n\n"
        "Rate your shows 1–10 to weight the results toward your favourites."
    )

# ─────────────────────────────────────────────
# Main — watched list builder
# ─────────────────────────────────────────────
if dataset_loaded:
    st.subheader("📺 What have you watched?")
    st.write("Search for titles and add them to your watched list.")

    # Search box
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Search for a title…", placeholder="e.g. Attack on Titan")
    with col2:
        st.write("")  # spacer
        search_clicked = st.button("🔍 Search", use_container_width=True)

    if query and search_clicked:
        results = rec.search(query, n=8)
        if results.empty:
            st.warning("No titles found. Try a different spelling.")
        else:
            st.write("**Search results** — click ➕ to add to your watched list:")
            for _, row in results.iterrows():
                c1, c2, c3 = st.columns([4, 2, 1])
                with c1:
                    st.write(f"**{row['title']}** ({int(row['year']) if pd.notna(row['year']) else '?'})")
                with c2:
                    badge = {"Movie": "🎬", "TV Show": "📺", "Anime": "⛩️"}.get(row["type"], "🎞️")
                    st.write(f"{badge} {row['type']}")
                with c3:
                    if st.button("➕", key=f"add_{row['title']}"):
                        if "watched" not in st.session_state:
                            st.session_state["watched"] = {}
                        st.session_state["watched"][row["title"]] = 7  # default rating

    st.markdown("---")

    # Watched list with ratings
    if "watched" not in st.session_state:
        st.session_state["watched"] = {}

    if st.session_state["watched"]:
        st.subheader(f"✅ Your watched list ({len(st.session_state['watched'])} titles)")
        st.write("Drag the sliders to rate each title — higher ratings influence results more.")

        to_remove = []
        for title, rating in st.session_state["watched"].items():
            c1, c2, c3 = st.columns([3, 2, 1])
            with c1:
                st.write(f"**{title}**")
            with c2:
                new_rating = st.slider(
                    "Rating", 1, 10, rating, key=f"rating_{title}", label_visibility="collapsed"
                )
                st.session_state["watched"][title] = new_rating
            with c3:
                if st.button("🗑️", key=f"remove_{title}"):
                    to_remove.append(title)

        for t in to_remove:
            del st.session_state["watched"][t]
            st.rerun()

        st.markdown("---")

        # Get recommendations
        if st.button("✨ Get Recommendations", type="primary", use_container_width=True):
            with st.spinner("Finding your next watch…"):
                recs = rec.recommend(
                    watched=list(st.session_state["watched"].keys()),
                    ratings=st.session_state["watched"],
                    n=n_recs,
                    filter_type=filter_type,
                )

            if recs.empty:
                st.warning("Couldn't generate recommendations. Try adding more titles to your list.")
            else:
                st.subheader(f"🎯 Your Top {len(recs)} Recommendations")
                type_icons = {"Movie": "🎬", "TV Show": "📺", "Anime": "⛩️"}

                for i, row in recs.iterrows():
                    icon = type_icons.get(row["type"], "🎞️")
                    with st.container():
                        c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
                        with c1:
                            st.markdown(f"**{i+1}. {row['title']}**")
                            st.caption(row["genres"])
                        with c2:
                            st.write(f"{icon} {row['type']}")
                        with c3:
                            year = int(row["year"]) if pd.notna(row["year"]) else "?"
                            st.write(f"⭐ {row['score']:.1f}  |  📅 {year}")
                        with c4:
                            match_pct = int(row["final_score"] * 100)
                            st.write(f"**{match_pct}%** match")
                        st.divider()
    else:
        st.info("👆 Search for titles above and add them to your watched list to get started.")
