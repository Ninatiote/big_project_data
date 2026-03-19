"""
app.py
------
Amazon Product Recommendation System
TBS Education — Data & AI Project · Group 4

Run with:
    pip install streamlit pandas numpy scikit-learn openpyxl
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Amazon Recommender · TBS Group 4",
    page_icon="🛍️",
    layout="wide"
)


# ══════════════════════════════════════════════════════════
# AMAZON-THEMED CSS
# ══════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* ── Amazon color palette ── */
    :root {
        --amazon-orange  : #FF9900;
        --amazon-dark    : #131921;
        --amazon-navy    : #232F3E;
        --amazon-blue    : #37475A;
        --amazon-light   : #FEBD69;
        --amazon-bg      : #EAEDED;
        --amazon-white   : #FFFFFF;
        --amazon-text    : #0F1111;
        --amazon-muted   : #565959;
        --amazon-green   : #007600;
        --amazon-red     : #B12704;
    }

    /* ── Hide Streamlit default chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 0 !important; }

    /* ── App background ── */
    .stApp { background-color: var(--amazon-bg) !important; }

    /* ── Top navigation bar ── */
    .amazon-navbar {
        background: var(--amazon-dark);
        padding: 10px 24px;
        display: flex;
        align-items: center;
        gap: 20px;
        margin-bottom: 16px;
    }
    .amazon-logo {
        font-size: 22px;
        font-weight: 700;
        color: var(--amazon-white);
        letter-spacing: -0.5px;
    }
    .amazon-logo span { color: var(--amazon-orange); }
    .amazon-tagline {
        font-size: 12px;
        color: var(--amazon-bg);
        opacity: 0.7;
    }
    .nav-badge {
        margin-left: auto;
        background: var(--amazon-orange);
        color: var(--amazon-dark);
        font-size: 11px;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 12px;
        letter-spacing: 0.03em;
    }

    /* ── KPI cards ── */
    .kpi-row {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 10px;
        margin-bottom: 16px;
    }
    .kpi-card {
        background: var(--amazon-white);
        border: 1px solid #D5D9D9;
        border-radius: 8px;
        padding: 14px 16px;
        border-top: 3px solid var(--amazon-orange);
    }
    .kpi-label {
        font-size: 11px;
        color: var(--amazon-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 22px;
        font-weight: 700;
        color: var(--amazon-text);
    }
    .kpi-sub {
        font-size: 11px;
        color: var(--amazon-green);
        margin-top: 2px;
    }

    /* ── Section header ── */
    .section-header {
        background: var(--amazon-navy);
        color: var(--amazon-white);
        padding: 10px 16px;
        border-radius: 8px 8px 0 0;
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 0;
    }
    .section-body {
        background: var(--amazon-white);
        border: 1px solid #D5D9D9;
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 16px;
        margin-bottom: 16px;
    }

    /* ── Recommendation cards ── */
    .rec-card {
        background: var(--amazon-white);
        border: 1px solid #D5D9D9;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 14px;
        transition: border-color 0.15s;
    }
    .rec-card:hover { border-color: var(--amazon-orange); }
    .rec-rank {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background: var(--amazon-orange);
        color: var(--amazon-dark);
        font-weight: 700;
        font-size: 13px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .rec-name {
        font-size: 14px;
        font-weight: 600;
        color: #007185;
    }
    .rec-name:hover { color: var(--amazon-red); text-decoration: underline; }
    .rec-stars { color: var(--amazon-orange); font-size: 13px; }
    .rec-score {
        margin-left: auto;
        background: #FFF3CD;
        color: #7D5A00;
        font-size: 12px;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 12px;
        white-space: nowrap;
        flex-shrink: 0;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--amazon-navy);
        border-radius: 8px 8px 0 0;
        gap: 2px;
        padding: 6px 8px 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: var(--amazon-blue) !important;
        color: var(--amazon-white) !important;
        border-radius: 6px 6px 0 0 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        padding: 8px 18px !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--amazon-orange) !important;
        color: var(--amazon-dark) !important;
        font-weight: 700 !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background: var(--amazon-white);
        border: 1px solid #D5D9D9;
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 20px !important;
    }

    /* ── Inputs ── */
    .stTextInput input {
        border: 1px solid #D5D9D9 !important;
        border-radius: 4px !important;
        padding: 8px 12px !important;
        font-size: 14px !important;
    }
    .stTextInput input:focus {
        border-color: var(--amazon-orange) !important;
        box-shadow: 0 0 0 2px rgba(255,153,0,0.3) !important;
    }
    .stSelectbox > div > div {
        border: 1px solid #D5D9D9 !important;
        border-radius: 4px !important;
    }

    /* ── Primary button → Amazon orange ── */
    .stButton > button[kind="primary"] {
        background: var(--amazon-orange) !important;
        color: var(--amazon-dark) !important;
        border: 1px solid #C45500 !important;
        border-radius: 20px !important;
        font-weight: 700 !important;
        padding: 8px 22px !important;
        font-size: 14px !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--amazon-light) !important;
    }
    .stButton > button {
        border-radius: 20px !important;
        font-size: 14px !important;
    }

    /* ── Metrics ── */
    [data-testid="metric-container"] {
        background: var(--amazon-white);
        border: 1px solid #D5D9D9;
        border-top: 3px solid var(--amazon-orange);
        border-radius: 8px;
        padding: 12px 16px !important;
    }
    [data-testid="metric-container"] label {
        color: var(--amazon-muted) !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: var(--amazon-text) !important;
        font-weight: 700 !important;
    }

    /* ── Dataframe ── */
    .stDataFrame { border: 1px solid #D5D9D9; border-radius: 6px; overflow: hidden; }
    .stDataFrame thead th {
        background: var(--amazon-navy) !important;
        color: white !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: #F0F2F2 !important;
        border: 1px solid #D5D9D9 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }

    /* ── Info / warning / success boxes ── */
    .stAlert { border-radius: 6px !important; }

    /* ── Sidebar (if used) ── */
    .css-1d391kg { background: var(--amazon-navy) !important; }

    /* ── Similar product similarity bar ── */
    .sim-bar-bg {
        background: #EAEDED;
        border-radius: 4px;
        height: 8px;
        width: 100%;
        margin-top: 4px;
    }
    .sim-bar-fill {
        height: 100%;
        border-radius: 4px;
        background: var(--amazon-orange);
    }

    /* ── User history pill ── */
    .hist-pill {
        display: inline-block;
        background: #FFF3CD;
        color: #7D5A00;
        border: 1px solid #FEBD69;
        border-radius: 14px;
        padding: 3px 10px;
        font-size: 12px;
        margin: 3px 4px 3px 0;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    # ── Adaptez le nom du fichier si nécessaire ──────────
    df = pd.read_excel("Group4 (1).xlsx")

    df = df.dropna(subset=["UserId", "ProductId", "Rating"])
    df = df.drop_duplicates(subset=["UserId", "ProductId"])
    df["Rating"]       = df["Rating"].astype(int)
    df["product_name"] = df["product_name"].str.strip()

    MIN = 5
    user_counts    = df["UserId"].value_counts()
    product_counts = df["ProductId"].value_counts()
    df_f = df[
        df["UserId"].isin(user_counts[user_counts >= MIN].index) &
        df["ProductId"].isin(product_counts[product_counts >= MIN].index)
    ].copy()

    product_meta = (
        df_f[["ProductId", "product_name"]]
        .drop_duplicates("ProductId")
        .set_index("ProductId")
    )

    matrix = df_f.pivot_table(
        index="UserId", columns="ProductId", values="Rating"
    ).fillna(0)

    user_sim = pd.DataFrame(
        cosine_similarity(matrix),
        index=matrix.index, columns=matrix.index
    )
    item_sim = pd.DataFrame(
        cosine_similarity(matrix.T),
        index=matrix.columns, columns=matrix.columns
    )

    return df, df_f, product_meta, matrix, user_sim, item_sim


# ══════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════

def popularity_recommender(user_id, df_f, product_meta, matrix, n=5):
    already = (
        matrix.loc[user_id][matrix.loc[user_id] > 0].index.tolist()
        if user_id in matrix.index else []
    )
    pop = df_f.groupby("ProductId").agg(
        avg_rating=("Rating", "mean"),
        nb_ratings=("Rating", "count")
    )
    pop = pop[pop["nb_ratings"] >= 5].copy()
    pop["score"] = pop["avg_rating"] * np.log1p(pop["nb_ratings"])
    recs = (
        pop[~pop.index.isin(already)]
        .sort_values("score", ascending=False)
        .head(n)
        .join(product_meta)
    )
    recs["avg_rating"] = recs["avg_rating"].round(2)
    recs["score"]      = recs["score"].round(2)
    return recs.reset_index()[["product_name", "avg_rating", "nb_ratings", "score"]]


def user_based_recommender(user_id, matrix, user_sim, product_meta, n=5, k=20):
    if user_id not in user_sim.index:
        return None
    scores    = user_sim[user_id].drop(user_id).sort_values(ascending=False)
    neighbors = scores.head(k).index.tolist()
    already   = matrix.loc[user_id][matrix.loc[user_id] > 0].index.tolist()
    preds = {}
    for product in matrix.columns:
        if product in already:
            continue
        r_list, s_list = [], []
        for nb in neighbors:
            r = matrix.loc[nb, product]
            if r > 0:
                r_list.append(r)
                s_list.append(scores[nb])
        if r_list and sum(s_list) > 0:
            preds[product] = np.dot(s_list, r_list) / sum(s_list)
    if not preds:
        return None
    result = (
        pd.Series(preds).sort_values(ascending=False).head(n)
        .to_frame("predicted_rating")
        .join(product_meta)
    )
    result["predicted_rating"] = result["predicted_rating"].round(2)
    return result.reset_index()[["product_name", "predicted_rating"]]


def item_based_recommender(user_id, matrix, item_sim, product_meta, n=5):
    if user_id not in matrix.index:
        return None
    rated   = matrix.loc[user_id]
    rated   = rated[rated > 0]
    already = rated.index.tolist()
    preds = {}
    for product in matrix.columns:
        if product in already:
            continue
        sims = item_sim.loc[product, already]
        if sims.sum() > 0:
            preds[product] = np.dot(sims.values, rated.values) / sims.sum()
    if not preds:
        return None
    result = (
        pd.Series(preds).sort_values(ascending=False).head(n)
        .to_frame("predicted_rating")
        .join(product_meta)
    )
    result["predicted_rating"] = result["predicted_rating"].round(2)
    return result.reset_index()[["product_name", "predicted_rating"]]


def get_similar_products(product_id, item_sim, product_meta, n=5):
    if product_id not in item_sim.index:
        return None
    sims = item_sim[product_id].drop(product_id).sort_values(ascending=False).head(n)
    result = sims.to_frame("similarity").join(product_meta)
    result["similarity"] = result["similarity"].round(3)
    return result.reset_index()[["product_name", "similarity"]]


# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════

def star_html(rating):
    full  = int(round(float(rating)))
    full  = max(0, min(5, full))
    return "★" * full + "☆" * (5 - full)


def render_rec_card(rank, name, score, label="Score"):
    stars = star_html(score)
    st.markdown(f"""
    <div class="rec-card">
        <div class="rec-rank">{rank}</div>
        <div style="flex:1;min-width:0">
            <div class="rec-name">{name[:65]}</div>
            <div class="rec-stars" style="margin-top:3px">{stars}</div>
        </div>
        <div class="rec-score">{label}: {score}</div>
    </div>
    """, unsafe_allow_html=True)


def render_similar_card(rank, name, sim):
    pct   = int(sim * 100)
    color = "#FF9900" if pct > 70 else "#FEBD69" if pct > 40 else "#D5D9D9"
    st.markdown(f"""
    <div class="rec-card">
        <div class="rec-rank">{rank}</div>
        <div style="flex:1;min-width:0">
            <div class="rec-name">{name[:65]}</div>
            <div class="sim-bar-bg">
                <div class="sim-bar-fill" style="width:{pct}%;background:{color}"></div>
            </div>
            <div style="font-size:11px;color:#565959;margin-top:2px">Similarity: {sim}</div>
        </div>
        <div class="rec-score">{pct}%</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════

df, df_f, product_meta, matrix, user_sim, item_sim = load_data()

n_users    = df["UserId"].nunique()
n_products = df["ProductId"].nunique()
n_ratings  = len(df)
avg_rating = round(df["Rating"].mean(), 2)
sparsity   = round((1 - n_ratings / (n_users * n_products)) * 100, 2)


# ══════════════════════════════════════════════════════════
# NAVBAR
# ══════════════════════════════════════════════════════════

st.markdown(f"""
<div class="amazon-navbar">
    <div>
        <div class="amazon-logo"><span>amazon</span> recommender</div>
        <div class="amazon-tagline">Beauty & Personal Care · Powered by Collaborative Filtering</div>
    </div>
    <div class="nav-badge">TBS Education · Group 4</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# KPIs
# ══════════════════════════════════════════════════════════

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Ratings",    f"{n_ratings:,}")
c2.metric("Users",            f"{n_users:,}")
c3.metric("Products",         f"{n_products:,}")
c4.metric("Avg Rating",       f"{avg_rating} ★")
c5.metric("Matrix Sparsity",  f"{sparsity}%")

st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "👤  User Recommendations",
    "🔗  Similar Products",
    "📊  Data Exploration"
])


# ──────────────────────────────────────────────────────────
# TAB 1 — User Recommendations
# ──────────────────────────────────────────────────────────

with tab1:
    st.markdown("### 👤 Personalized Recommendations")
    st.write("Enter a User ID to get product recommendations based on their rating history.")

    col_id, col_method, col_n = st.columns([3, 2, 1])
    with col_id:
        user_id = st.text_input(
            "User ID",
            placeholder="e.g. AG73BVBKUOH22USSFJA5ZWL7AKXA"
        )
    with col_method:
        method = st.selectbox(
            "Recommendation method",
            ["User-Based CF", "Item-Based CF", "Popularity-Based", "Compare all 3"]
        )
    with col_n:
        n_recs = st.slider("# Recs", min_value=3, max_value=10, value=5)

    if st.button("🔍 Get Recommendations", type="primary"):
        if not user_id.strip():
            st.warning("⚠️ Please enter a User ID.")
        else:
            uid = user_id.strip()

            # User history
            history = (
                df_f[df_f["UserId"] == uid][["product_name", "Rating"]]
                .sort_values("Rating", ascending=False)
            )
            if history.empty:
                st.info("ℹ️ User not found — showing Popularity-Based recommendations.")
            else:
                pills = "".join([
                    f'<span class="hist-pill">{"★"*int(r["Rating"])} {r["product_name"][:30]}</span>'
                    for _, r in history.head(5).iterrows()
                ])
                st.markdown(
                    f'<div style="margin-bottom:12px"><b>Rating history ({len(history)} products):</b><br>{pills}</div>',
                    unsafe_allow_html=True
                )

            st.markdown("---")

            if method == "Compare all 3":
                st.markdown("#### Side-by-side comparison")
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    st.markdown("**📊 Popularity-Based**")
                    r = popularity_recommender(uid, df_f, product_meta, matrix, n_recs)
                    for i, row in r.iterrows():
                        render_rec_card(i+1, row["product_name"], row["avg_rating"], "Avg ★")

                with col_b:
                    st.markdown("**👥 User-Based CF**")
                    r = user_based_recommender(uid, matrix, user_sim, product_meta, n_recs)
                    if r is not None:
                        for i, row in r.iterrows():
                            render_rec_card(i+1, row["product_name"], row["predicted_rating"], "Pred ★")
                    else:
                        st.info("Not enough data.")

                with col_c:
                    st.markdown("**🛒 Item-Based CF**")
                    r = item_based_recommender(uid, matrix, item_sim, product_meta, n_recs)
                    if r is not None:
                        for i, row in r.iterrows():
                            render_rec_card(i+1, row["product_name"], row["predicted_rating"], "Pred ★")
                    else:
                        st.info("Not enough data.")

            elif method == "Popularity-Based":
                st.markdown("#### 📊 Popularity-Based Recommendations")
                r = popularity_recommender(uid, df_f, product_meta, matrix, n_recs)
                for i, row in r.iterrows():
                    render_rec_card(i+1, row["product_name"], row["avg_rating"], "Avg ★")

            elif method == "User-Based CF":
                st.markdown("#### 👥 User-Based Collaborative Filtering")
                r = user_based_recommender(uid, matrix, user_sim, product_meta, n_recs)
                if r is not None:
                    for i, row in r.iterrows():
                        render_rec_card(i+1, row["product_name"], row["predicted_rating"], "Pred ★")
                else:
                    st.warning("User not found — showing Popularity-Based instead.")
                    r = popularity_recommender(uid, df_f, product_meta, matrix, n_recs)
                    for i, row in r.iterrows():
                        render_rec_card(i+1, row["product_name"], row["avg_rating"], "Avg ★")

            elif method == "Item-Based CF":
                st.markdown("#### 🛒 Item-Based Collaborative Filtering")
                r = item_based_recommender(uid, matrix, item_sim, product_meta, n_recs)
                if r is not None:
                    for i, row in r.iterrows():
                        render_rec_card(i+1, row["product_name"], row["predicted_rating"], "Pred ★")
                else:
                    st.warning("User not found — showing Popularity-Based instead.")
                    r = popularity_recommender(uid, df_f, product_meta, matrix, n_recs)
                    for i, row in r.iterrows():
                        render_rec_card(i+1, row["product_name"], row["avg_rating"], "Avg ★")

    # Sample IDs
    with st.expander("💡 Sample User IDs to test"):
        st.markdown("""
        | User ID | Ratings | Avg Rating |
        |---|---|---|
        | `AG73BVBKUOH22USSFJA5ZWL7AKXA` | 69 | 4.6 ★ |
        | `AEZP6Z2C5AVQDZAJECQYZWQRNG3Q` | 48 | 4.2 ★ |
        | `AEMP3A7IKW37CMWFXNKXWW6HGJHA_1` | 45 | 3.9 ★ |
        | `AGZUJTI7A3JFKB4FP5JOH6NVAJIQ_1` | 35 | 4.7 ★ |
        | `AEHWKRPNWNMOAJSMO2F6O7RFRTNA` | 27 | 5.0 ★ |
        """)


# ──────────────────────────────────────────────────────────
# TAB 2 — Similar Products
# ──────────────────────────────────────────────────────────

with tab2:
    st.markdown("### 🔗 Find Similar Products")
    st.write("Search for a product to discover similar ones based on user rating patterns.")

    search = st.text_input(
        "Search product",
        placeholder="e.g. Acne Pimple Patch",
        key="search_item"
    )
    n_sim = st.slider("# Similar products", min_value=3, max_value=10, value=5)

    if search:
        matches = product_meta[
            product_meta["product_name"].str.contains(search, case=False, na=False)
        ]

        if matches.empty:
            st.warning("❌ No product found. Try a different keyword.")
        else:
            selected_pid = st.selectbox(
                f"🔎 {len(matches)} product(s) found — select one:",
                options=matches.index.tolist(),
                format_func=lambda x: product_meta.loc[x, "product_name"][:70]
            )

            if st.button("🔗 Find similar products", type="primary"):
                prod_data = df_f[df_f["ProductId"] == selected_pid]

                ca, cb, cc = st.columns(3)
                ca.metric("Ratings",    len(prod_data))
                cb.metric("Avg Rating", f"{prod_data['Rating'].mean():.2f} ★" if not prod_data.empty else "N/A")
                cc.metric("Product ID", selected_pid[:16] + "...")

                st.markdown("---")

                similars = get_similar_products(selected_pid, item_sim, product_meta, n_sim)
                if similars is not None and not similars.empty:
                    st.markdown("#### Most similar products")
                    for i, row in similars.iterrows():
                        render_similar_card(i+1, row["product_name"], row["similarity"])

                    st.markdown("#### Similarity scores")
                    st.bar_chart(similars.set_index("product_name")["similarity"])
                else:
                    st.warning("This product is not in the similarity matrix.")


# ──────────────────────────────────────────────────────────
# TAB 3 — Data Exploration
# ──────────────────────────────────────────────────────────

with tab3:
    st.markdown("### 📊 Data Exploration")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Rating distribution**")
        rating_dist = df["Rating"].value_counts().sort_index().reset_index()
        rating_dist.columns = ["Rating", "Count"]
        st.bar_chart(rating_dist.set_index("Rating"))

    with col_r:
        st.markdown("**Top 10 most rated products**")
        top10 = (
            df_f.groupby("ProductId")
            .agg(nb_ratings=("Rating", "count"), avg_rating=("Rating", "mean"))
            .sort_values("nb_ratings", ascending=False)
            .head(10)
            .join(product_meta)
        )
        top10["avg_rating"]   = top10["avg_rating"].round(2)
        top10["product_name"] = top10["product_name"].str[:40]
        st.dataframe(
            top10[["product_name", "nb_ratings", "avg_rating"]].reset_index(drop=True),
            use_container_width=True
        )

    st.markdown("---")

    st.markdown("**Top 10 most active users**")
    top_users = (
        df_f.groupby("UserId")
        .agg(nb_ratings=("Rating", "count"), avg_rating=("Rating", "mean"))
        .sort_values("nb_ratings", ascending=False)
        .head(10)
        .reset_index()
    )
    top_users["avg_rating"] = top_users["avg_rating"].round(2)
    st.dataframe(top_users, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.markdown("**Dataset summary**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        - Total ratings : **{n_ratings:,}**
        - Unique users : **{n_users:,}**
        - Unique products : **{n_products:,}**
        """)
    with col2:
        st.markdown(f"""
        - Average rating : **{avg_rating} / 5**
        - Matrix sparsity : **{sparsity}%**
        - Filtered users (>= 5 ratings) : **{df_f["UserId"].nunique():,}**
        """)