"""
Media Recommender — Netflix-style Streamlit UI
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
from recommender import MediaRecommender, load_data

st.set_page_config(page_title="WatchNext", layout="wide", initial_sidebar_state="collapsed")

# ── Theme ──
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

dark = st.session_state.dark_mode

if dark:
    bg         = "#0d0d0d"
    surface    = "#161616"
    border     = "#2a2a2a"
    text       = "#f0f0f0"
    text_muted = "#888888"
    accent     = "#e50914"
    acc_hover  = "#ff1a24"
    tag_bg     = "#2a2a2a"
    tag_text   = "#cccccc"
    input_bg   = "#1a1a1a"
    card_bg    = "#161616"
else:
    bg         = "#f4f4f4"
    surface    = "#ffffff"
    border     = "#e0e0e0"
    text       = "#111111"
    text_muted = "#666666"
    accent     = "#e50914"
    acc_hover  = "#c0070f"
    tag_bg     = "#eeeeee"
    tag_text   = "#444444"
    input_bg   = "#ffffff"
    card_bg    = "#ffffff"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

.stApp {{ background: {bg} !important; font-family: 'DM Sans', sans-serif; color: {text}; }}
#MainMenu, footer, header {{ visibility: hidden !important; }}
.block-container {{ padding: 0 !important; max-width: 100% !important; }}
section[data-testid="stSidebar"] {{ display: none !important; }}
div[data-testid="stDecoration"] {{ display: none !important; }}

.wn-logo {{ font-family: 'Bebas Neue', sans-serif; font-size: 1.9rem; color: {accent}; letter-spacing: 3px; }}
.wn-tagline {{ font-size: 0.7rem; color: {text_muted}; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 2px; }}
.wn-hero-title {{ font-family: 'Bebas Neue', sans-serif; font-size: clamp(2.2rem, 4vw, 3.6rem); color: {text}; letter-spacing: 1px; line-height: 1.05; }}
.wn-hero-sub {{ color: {text_muted}; font-size: 0.95rem; font-weight: 300; margin-top: 6px; }}

.wn-section-label {{
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: {text_muted}; margin-bottom: 12px; margin-top: 4px;
}}

.wn-stat-row {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 24px; }}
.wn-stat {{
    padding: 5px 13px; border-radius: 20px; background: {tag_bg};
    font-size: 0.73rem; color: {tag_text}; font-weight: 500; border: 1px solid {border};
}}

.wn-result {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 11px 14px; border-radius: 8px; border: 1px solid {border};
    background: {surface}; margin-bottom: 8px; transition: border-color 0.15s, transform 0.15s;
}}
.wn-result:hover {{ border-color: {accent}; transform: translateX(3px); }}
.wn-result-left {{ flex: 1; min-width: 0; }}
.wn-result-title {{ font-weight: 500; font-size: 0.9rem; color: {text}; }}
.wn-result-meta {{ font-size: 0.75rem; color: {text_muted}; margin-top: 3px; }}
.wn-badge {{
    display: inline-block; padding: 2px 7px; border-radius: 3px;
    font-size: 0.64rem; font-weight: 700; letter-spacing: 0.07em;
    text-transform: uppercase; background: {tag_bg}; color: {tag_text}; margin-left: 7px;
    vertical-align: middle;
}}

.wn-rec-card {{
    padding: 14px 16px; border-radius: 10px; border: 1px solid {border};
    background: {card_bg}; margin-bottom: 10px;
    transition: transform 0.15s, border-color 0.15s;
    position: relative; overflow: hidden;
}}
.wn-rec-card:hover {{ transform: translateX(5px); border-color: {accent}; }}
.wn-rec-card::before {{
    content: ''; position: absolute; left: 0; top: 0; bottom: 0;
    width: 3px; background: {accent}; opacity: 0; transition: opacity 0.15s;
}}
.wn-rec-card:hover::before {{ opacity: 1; }}
.wn-rec-num {{
    font-family: 'Bebas Neue', sans-serif; font-size: 2.2rem; line-height: 1;
    color: {border}; float: left; margin-right: 14px; margin-top: -3px;
}}
.wn-rec-body {{ overflow: hidden; }}
.wn-rec-title {{ font-weight: 600; font-size: 0.92rem; color: {text}; }}
.wn-rec-meta {{ font-size: 0.75rem; color: {text_muted}; margin-top: 3px; }}
.wn-rec-genres {{ font-size: 0.71rem; color: {text_muted}; font-style: italic; margin-top: 5px; }}
.wn-match-row {{ display: flex; align-items: center; gap: 8px; margin-top: 9px; }}
.wn-match-bg {{ flex: 1; height: 3px; background: {border}; border-radius: 2px; overflow: hidden; }}
.wn-match-fill {{ height: 3px; border-radius: 2px; background: linear-gradient(90deg, {accent}, #ff6b6b); }}
.wn-match-pct {{ font-size: 0.7rem; font-weight: 700; color: {accent}; white-space: nowrap; }}

.wn-watched-item {{ padding: 10px 0; border-bottom: 1px solid {border}; }}
.wn-watched-title {{ font-weight: 500; font-size: 0.88rem; color: {text}; margin-bottom: 4px; }}
.wn-watched-rating {{ font-size: 0.72rem; color: {accent}; font-weight: 600; margin-top: 2px; }}

.wn-empty {{
    text-align: center; padding: 40px 20px;
    color: {text_muted}; font-size: 0.87rem; line-height: 1.7;
}}
.wn-empty-glyph {{ font-family: 'Bebas Neue', sans-serif; font-size: 3rem; color: {border}; margin-bottom: 8px; }}

/* Input overrides */
div[data-testid="stTextInput"] input {{
    background: {input_bg} !important; border: 1px solid {border} !important;
    border-radius: 8px !important; color: {text} !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important; padding: 10px 14px !important;
}}
div[data-testid="stTextInput"] input:focus {{
    border-color: {accent} !important; box-shadow: 0 0 0 2px {accent}20 !important;
    outline: none !important;
}}
div[data-testid="stButton"] button[kind="primary"] {{
    background: {accent} !important; color: #fff !important; border: none !important;
    border-radius: 8px !important; font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important; letter-spacing: 0.04em !important;
    transition: background 0.15s !important;
}}
div[data-testid="stButton"] button[kind="primary"]:hover {{ background: {acc_hover} !important; }}
div[data-testid="stButton"] button:not([kind="primary"]) {{
    background: transparent !important; border: 1px solid {border} !important;
    color: {text_muted} !important; border-radius: 6px !important;
    font-size: 0.78rem !important; font-family: 'DM Sans', sans-serif !important;
}}
div[data-testid="stButton"] button:not([kind="primary"]):hover {{
    border-color: {accent} !important; color: {accent} !important;
}}
div[data-testid="stSelectbox"] > div > div {{
    background: {input_bg} !important; border: 1px solid {border} !important;
    border-radius: 8px !important; color: {text} !important;
}}
div[data-testid="stSlider"] div[role="slider"] {{ background: {accent} !important; border-color: {accent} !important; }}

hr {{ border: none !important; border-top: 1px solid {border} !important; margin: 18px 0 !important; }}
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: {bg}; }}
::-webkit-scrollbar-thumb {{ background: {border}; border-radius: 2px; }}
</style>
""", unsafe_allow_html=True)

# ── Load data ──
@st.cache_resource(show_spinner="Loading recommendation engine...")
def get_recommender():
    df = load_data(
        movies_path="data/movies.csv",
        shows_path="data/shows.csv",
        anime_path="data/anime.csv",
    )
    rec = MediaRecommender()
    rec.fit(df)
    return rec

try:
    rec = get_recommender()
    counts = rec.df["type"].value_counts().to_dict()
    total  = len(rec.df)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ── Session state ──
for key, default in [("watched", {}), ("recs", None), ("_last_query", "")]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
# NAV BAR
# ─────────────────────────────────────────────
n1, n2, n3, n4 = st.columns([3, 2, 1, 1])
with n1:
    st.markdown(f'''
    <div style="padding: 20px 48px 0;">
        <div class="wn-logo">WatchNext</div>
        <div class="wn-tagline">Movies &nbsp;&middot;&nbsp; TV Shows &nbsp;&middot;&nbsp; Anime</div>
    </div>''', unsafe_allow_html=True)
with n3:
    st.write("")
    filter_type = st.selectbox("Type filter", ["All", "Movie", "TV Show", "Anime"],
                               label_visibility="collapsed")
    filter_type = None if filter_type == "All" else filter_type
with n4:
    st.write("")
    mode_label = "Light" if dark else "Dark"
    if st.button(mode_label + " mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

st.markdown(f'<hr style="margin:10px 0;">', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown(f'''
<div style="padding: 40px 48px 24px;">
    <div class="wn-hero-title">Your next obsession<br>is one search away.</div>
    <div class="wn-hero-sub">Search anything you have watched. We will find what you will love next.</div>
</div>''', unsafe_allow_html=True)

stats_html = '<div class="wn-stat-row" style="padding: 0 48px 24px;">'
for t, n in counts.items():
    stats_html += f'<div class="wn-stat">{n:,} {t}s</div>'
stats_html += f'<div class="wn-stat">{total:,} total</div></div>'
st.markdown(stats_html, unsafe_allow_html=True)

st.markdown(f'<hr style="margin: 0 0 0 0;">', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN — two columns
# ─────────────────────────────────────────────
left_pad, left_col, mid_pad, right_col, right_pad = st.columns([1, 10, 1, 7, 1])

with left_col:
    # ── Search ──
    st.write("")
    st.markdown('<div class="wn-section-label">Search</div>', unsafe_allow_html=True)

    sc1, sc2 = st.columns([5, 1])
    with sc1:
        query = st.text_input("search", placeholder="Search movies, shows, anime...",
                               label_visibility="collapsed")
    with sc2:
        st.write("")
        search_btn = st.button("Search", use_container_width=True)

    # Trigger on Enter (query changed) OR button click
    do_search = search_btn or (query and query != st.session_state._last_query)

    if do_search and query:
        st.session_state._last_query = query
        results = rec.search(query, n=8)

        if results.empty:
            st.warning("No titles found — try a different spelling.")
        else:
            st.markdown('<div class="wn-section-label" style="margin-top:20px;">Results</div>', unsafe_allow_html=True)
            for _, row in results.iterrows():
                already = row["title"] in st.session_state.watched
                year  = int(row["year"]) if pd.notna(row.get("year", float("nan"))) else "?"
                score = f'{row["score"]:.1f}' if row.get("score") else "?"
                genres = (row["genres"] or "")[:45]

                rc1, rc2 = st.columns([6, 1])
                with rc1:
                    st.markdown(f'''
                    <div class="wn-result">
                        <div class="wn-result-left">
                            <div class="wn-result-title">
                                {row["title"]}<span class="wn-badge">{row["type"]}</span>
                            </div>
                            <div class="wn-result-meta">{year} &nbsp;&middot;&nbsp; {score}/10 &nbsp;&middot;&nbsp; {genres}</div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                with rc2:
                    if already:
                        st.markdown(f'<div style="color:{text_muted}; font-size:0.75rem; padding-top:14px; text-align:center;">Added</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.write("")
                        if st.button("+ Add", key=f"add_{row['title']}"):
                            st.session_state.watched[row["title"]] = 7
                            st.session_state.recs = None
                            st.rerun()

    # ── Recommendations ──
    if st.session_state.recs is not None:
        recs = st.session_state.recs
        st.markdown(f'<hr>', unsafe_allow_html=True)
        st.markdown('<div class="wn-section-label">Recommended for you</div>', unsafe_allow_html=True)

        if recs.empty:
            st.markdown('<div class="wn-empty"><div>No recommendations found. Try adding more titles or changing the type filter.</div></div>',
                        unsafe_allow_html=True)
        else:
            for i, row in recs.iterrows():
                year  = int(row["year"]) if pd.notna(row.get("year", float("nan"))) else "?"
                score = f'{row["score"]:.1f}' if row.get("score") else "?"
                pct   = min(int(row["final_score"] * 100), 99)
                genres = (row["genres"] or "")[:55]
                st.markdown(f'''
                <div class="wn-rec-card">
                    <div class="wn-rec-num">{i+1:02d}</div>
                    <div class="wn-rec-body">
                        <div class="wn-rec-title">{row["title"]} <span class="wn-badge">{row["type"]}</span></div>
                        <div class="wn-rec-meta">{year} &nbsp;&middot;&nbsp; {score}/10</div>
                        <div class="wn-rec-genres">{genres}</div>
                        <div class="wn-match-row">
                            <div class="wn-match-bg"><div class="wn-match-fill" style="width:{pct}%;"></div></div>
                            <div class="wn-match-pct">{pct}% match</div>
                        </div>
                    </div>
                    <div style="clear:both;"></div>
                </div>
                ''', unsafe_allow_html=True)

with right_col:
    # ── Watched list + get recs ──
    st.write("")
    st.markdown('<div class="wn-section-label">Your watched list</div>', unsafe_allow_html=True)

    n_recs = st.slider("How many recommendations?", 5, 25, 10)

    if not st.session_state.watched:
        st.markdown('''
        <div class="wn-empty">
            <div class="wn-empty-glyph">[ ]</div>
            <div>Search for titles on the left<br>and add them to get started.</div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        to_remove = []
        for title, rating in list(st.session_state.watched.items()):
            wc1, wc2 = st.columns([4, 1])
            with wc1:
                st.markdown(f'<div class="wn-watched-title">{title}</div>', unsafe_allow_html=True)
                new_rating = st.slider("r", 1, 10, rating, key=f"s_{title}",
                                       label_visibility="collapsed")
                st.session_state.watched[title] = new_rating
                st.markdown(f'<div class="wn-watched-rating">{new_rating} / 10</div>',
                            unsafe_allow_html=True)
            with wc2:
                st.write("")
                st.write("")
                if st.button("x", key=f"rm_{title}"):
                    to_remove.append(title)
            st.markdown(f'<hr style="margin: 6px 0;">', unsafe_allow_html=True)

        for t in to_remove:
            del st.session_state.watched[t]
            st.session_state.recs = None
            st.rerun()

        st.write("")
        if st.button("Get Recommendations", type="primary", use_container_width=True):
            with st.spinner("Finding your next watch..."):
                st.session_state.recs = rec.recommend(
                    watched=list(st.session_state.watched.keys()),
                    ratings=st.session_state.watched,
                    n=n_recs,
                    filter_type=filter_type,
                )
            st.rerun()
