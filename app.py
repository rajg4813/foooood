from pathlib import Path
import html
import math

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# NutriPlan AI — student-friendly Indian food recommender
# Dataset: Anuvaad Indian Nutrient Databank (INDB), Nov 2024
# ML: K-Nearest Neighbors over standardized nutrient features
# ------------------------------------------------------------

st.set_page_config(
    page_title="NutriPlan AI | Indian Food Planner",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "indb_foods.csv"

FEATURES = [
    "energy_kcal_100g",
    "carbohydrate_g_100g",
    "protein_g_100g",
    "fat_g_100g",
    "free_sugar_g_100g",
    "fibre_g_100g",
]

NUTRIENT_LABELS = {
    "energy_kcal_100g": "Calories",
    "carbohydrate_g_100g": "Carbs",
    "protein_g_100g": "Protein",
    "fat_g_100g": "Fat",
    "free_sugar_g_100g": "Free sugar",
    "fibre_g_100g": "Fibre",
}

@st.cache_data
def load_foods():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Keep the data folder beside app.py."
        )
    df = pd.read_csv(DATA_PATH)
    numeric_cols = FEATURES + [
        "serving_energy_kcal", "serving_carbohydrate_g",
        "serving_protein_g", "serving_fat_g",
        "serving_free_sugar_g", "serving_fibre_g",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["food_name"] = df["food_name"].fillna("Unnamed food").astype(str).str.strip()
    df["primary_source"] = df["primary_source"].fillna("Not specified").astype(str)
    return df

@st.cache_resource
def build_knn_model(frame):
    # Keep foods with complete nutrient features. This filters only model inputs;
    # the full dataset remains available in the Food Library and Dataset page.
    pool = frame.dropna(subset=FEATURES).copy()
    pool = pool.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    # Avoid recommending beverages as the main meal option.
    beverage_words = r"\b(tea|coffee|juice|drink|squash|syrup|water|soda)\b"
    pool = pool[~pool["food_name"].str.lower().str.contains(beverage_words, regex=True, na=False)]
    pool = pool[(pool["energy_kcal_100g"] >= 35) & (pool["energy_kcal_100g"] <= 650)]
    if len(pool) < 5:
        raise ValueError("Not enough suitable food rows to build the KNN recommender.")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(pool[FEATURES].astype(float))
    model = NearestNeighbors(metric="euclidean", algorithm="auto")
    model.fit(x_scaled)
    return pool.reset_index(drop=True), scaler, model

def estimate_daily_needs(age, sex, height_cm, weight_kg, activity, goal):
    # Mifflin-St Jeor equation: educational estimate, not a medical prescription.
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age
    if sex == "Male":
        bmr = base + 5
    elif sex == "Female":
        bmr = base - 161
    else:
        bmr = base - 78  # midpoint when user chooses not to specify
    activity_factor = {
        "Mostly sitting": 1.20,
        "Light activity (1–3 days/week)": 1.375,
        "Moderate activity (3–5 days/week)": 1.55,
        "Very active (6–7 days/week)": 1.725,
    }[activity]
    maintenance = bmr * activity_factor
    adjustment = {"Lose weight gently": -250, "Maintain weight": 0, "Build muscle / gain": 250}[goal]
    calories = max(0, maintenance + adjustment)
    protein = calories * 0.25 / 4
    carbs = calories * 0.45 / 4
    fat = calories * 0.30 / 9
    return {
        "bmr": bmr,
        "maintenance": maintenance,
        "calories": calories,
        "protein": protein,
        "carbs": carbs,
        "fat": fat,
    }

# Example nutrient-profile targets, per 100 g. These are heuristic search
# profiles for this academic prototype, not official dietary requirements.
PROFILE_TARGETS = {
    "Lose weight gently": {
        "Breakfast": [155, 18, 13, 5, 2.0, 5.0],
        "Lunch":     [190, 20, 17, 6, 2.0, 5.0],
        "Snack":     [125, 12, 11, 4, 1.5, 4.0],
        "Dinner":    [175, 17, 17, 6, 1.5, 5.0],
    },
    "Maintain weight": {
        "Breakfast": [205, 25, 12, 7, 3.0, 4.0],
        "Lunch":     [235, 28, 15, 8, 3.0, 4.0],
        "Snack":     [165, 18, 10, 5, 2.5, 3.0],
        "Dinner":    [215, 23, 15, 7, 2.5, 4.0],
    },
    "Build muscle / gain": {
        "Breakfast": [255, 29, 18, 8, 3.0, 4.0],
        "Lunch":     [285, 32, 21, 9, 3.0, 4.0],
        "Snack":     [225, 23, 16, 8, 3.0, 3.0],
        "Dinner":    [265, 27, 20, 9, 3.0, 4.0],
    },
}

def recommend_foods(pool, scaler, model, goal, meal, avoid_words, already_used, count=3):
    query = np.array(PROFILE_TARGETS[goal][meal], dtype=float).reshape(1, -1)
    query_scaled = scaler.transform(query)
    n = min(len(pool), max(count * 5, 20))
    distances, indices = model.kneighbors(query_scaled, n_neighbors=n)
    avoid = [w.strip().lower() for w in (avoid_words or "").split(",") if w.strip()]
    chosen = []
    for distance, idx in zip(distances[0], indices[0]):
        row = pool.iloc[int(idx)]
        name = row["food_name"].lower()
        code = str(row.get("food_code", ""))
        if code in already_used:
            continue
        if any(word in name for word in avoid):
            continue
        chosen.append((row, float(distance)))
        if len(chosen) >= count:
            break
    return chosen

def food_card(row, rank, distance=None):
    name = html.escape(str(row["food_name"]))
    source = html.escape(str(row["primary_source"]).replace("_", " "))
    kcal = float(row["energy_kcal_100g"])
    protein = float(row["protein_g_100g"])
    carbs = float(row["carbohydrate_g_100g"])
    fat = float(row["fat_g_100g"])
    fibre = float(row["fibre_g_100g"])
    sugar = float(row["free_sugar_g_100g"])
    st.markdown(
        f"""
        <div class="food-card">
          <div class="food-card-top">
            <span class="food-rank">0{rank}</span>
            <span class="food-source">{source}</span>
          </div>
          <div class="food-title">{name}</div>
          <div class="food-calories">{kcal:.0f} <span>kcal / 100 g</span></div>
          <div class="macro-grid">
            <div><b>{protein:.1f}g</b><small>PROTEIN</small></div>
            <div><b>{carbs:.1f}g</b><small>CARBS</small></div>
            <div><b>{fat:.1f}g</b><small>FAT</small></div>
            <div><b>{fibre:.1f}g</b><small>FIBRE</small></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def metric_tile(label, value, note=""):
    st.markdown(
        f"""<div class="metric-tile">
        <div class="metric-label">{html.escape(label)}</div>
        <div class="metric-value">{html.escape(str(value))}</div>
        <div class="metric-note">{html.escape(note)}</div>
        </div>""",
        unsafe_allow_html=True,
    )

# ------------------------------- Styling -------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#18352c; --green:#244e3d; --muted:#728078; --lime:#d9ef9c; --paper:#f5f7f1; }
    .stApp { background: #f5f7f1; color: var(--ink); font-family:'DM Sans',sans-serif; }
    [data-testid="stHeader"] { background: rgba(245,247,241,.94); }
    .block-container { max-width: 1220px; padding-top: 1.1rem; padding-bottom: 3rem; }
    h1,h2,h3 { color:#18352c !important; font-family:'Manrope','DM Sans',sans-serif !important; letter-spacing:-.04em; }
    p, label, .stMarkdown { color:#31483e; }
    div[data-testid="stRadio"] > div { gap:.5rem; }
    div[data-testid="stRadio"] label {
      background:#e9eee5; border:1px solid #dce5d9; border-radius:999px;
      padding: .45rem 1rem; margin-right: .15rem;
    }
    div[data-testid="stRadio"] label:has(input:checked) {
      background:#244e3d; border-color:#244e3d; color:#fff;
    }
    div[data-testid="stRadio"] label:has(input:checked) p { color:#fff !important; }
    .brandline { display:flex; align-items:center; gap:12px; margin: 0 0 1rem 0; }
    .brandmark { width:42px; height:42px; border-radius:14px; display:flex; align-items:center; justify-content:center;
      background:#244e3d; color:#e4f4ba; font-size:22px; box-shadow:0 8px 24px rgba(36,78,61,.15); }
    .brandname { font-family:'Manrope',sans-serif; font-weight:800; font-size:19px; letter-spacing:-.04em; color:#18352c; }
    .brandtag { color:#829087; text-transform:uppercase; letter-spacing:.16em; font-size:9px; font-weight:700; margin-top:1px; }
    .hero { background:linear-gradient(118deg,#183d30 0%,#245442 56%,#3d7055 100%); color:#f8fbf4; padding:42px 44px;
      border-radius:26px; position:relative; overflow:hidden; min-height:310px; box-shadow:0 20px 50px rgba(29,69,51,.14); }
    .hero:after { content:""; position:absolute; width:370px; height:370px; right:-75px; top:-120px; border:1px solid rgba(232,246,209,.15);
      border-radius:50%; box-shadow:0 0 0 36px rgba(232,246,209,.035),0 0 0 76px rgba(232,246,209,.025); }
    .hero-kicker { color:#d9ef9c; letter-spacing:.18em; text-transform:uppercase; font-size:10px; font-weight:700; margin-bottom:18px; }
    .hero-title { color:#fff; font-family:'Manrope',sans-serif; font-size:clamp(36px,5vw,58px); font-weight:800; line-height:1.04; letter-spacing:-.06em; margin:0; max-width:650px; }
    .hero-title span { color:#d9ef9c; }
    .hero-copy { max-width:520px; color:#d1e0d4; font-size:15px; line-height:1.65; margin-top:18px; }
    .hero-pill { display:inline-flex; border:1px solid rgba(228,244,186,.35); padding:8px 12px; border-radius:999px; color:#e5f1cf; font-size:11px; margin-top:20px; }
    .section-kicker { color:#73907f; font-size:10px; text-transform:uppercase; letter-spacing:.18em; font-weight:800; margin-bottom:6px; }
    .section-title { font-family:'Manrope',sans-serif; color:#18352c; font-weight:800; font-size:25px; letter-spacing:-.045em; margin-bottom:4px; }
    .section-sub { color:#738178; font-size:13px; margin-bottom:16px; }
    .metric-tile { background:#fff; border:1px solid #e5ebe1; border-radius:18px; padding:18px 20px; min-height:105px; box-shadow:0 7px 24px rgba(31,62,44,.035); }
    .metric-label { color:#7a897f; font-size:10px; text-transform:uppercase; letter-spacing:.13em; font-weight:800; }
    .metric-value { color:#1d4232; font-family:'Manrope',sans-serif; font-weight:800; font-size:27px; margin-top:5px; letter-spacing:-.05em; }
    .metric-note { color:#9aa59d; font-size:11px; margin-top:2px; }
    .food-card { border:1px solid #e5ebe1; border-radius:18px; background:#fff; padding:17px 17px 15px; min-height:190px; box-shadow:0 8px 28px rgba(34,65,47,.045); margin-bottom:10px; }
    .food-card-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; }
    .food-rank { color:#78917b; background:#eff5e8; border-radius:8px; padding:5px 8px; font-size:10px; font-weight:800; }
    .food-source { color:#9aa69d; text-transform:uppercase; letter-spacing:.08em; font-size:8px; font-weight:800; }
    .food-title { color:#1c3b2f; font-family:'Manrope',sans-serif; font-weight:800; font-size:16px; line-height:1.25; min-height:40px; }
    .food-calories { color:#315f47; font-weight:800; font-size:22px; margin:11px 0 12px; letter-spacing:-.04em; }
    .food-calories span { color:#8b988f; font-size:10px; font-weight:600; letter-spacing:0; }
    .macro-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:6px; border-top:1px solid #edf1ea; padding-top:11px; }
    .macro-grid div { display:flex; flex-direction:column; gap:2px; }
    .macro-grid b { color:#274a37; font-size:12px; }
    .macro-grid small { color:#a0aaa2; font-size:7px; letter-spacing:.1em; font-weight:800; }
    .info-strip { background:#edf4e8; border:1px solid #dfe9d9; color:#526d5a; border-radius:14px; padding:13px 16px; font-size:12px; line-height:1.6; }
    .dark-panel { background:#203f32; border-radius:20px; padding:22px 24px; color:#e9f2e6; }
    .dark-panel h3 { color:#f4f8ef !important; }
    .dark-panel p { color:#c4d7c7; }
    .stButton > button, .stFormSubmitButton > button { border-radius:999px; border:0; background:#244e3d; color:white; font-weight:700; padding:.62rem 1.2rem; }
    .stButton > button:hover, .stFormSubmitButton > button:hover { background:#35694f; color:white; border:0; }
    div[data-baseweb="input"] input, div[data-baseweb="select"] > div { border-radius:12px; }
    [data-testid="stDataFrame"] { border:1px solid #e4ebe1; border-radius:14px; overflow:hidden; }
    hr { border-color:#e3eae0; }
    .footer-note { color:#98a49a; font-size:10px; text-align:center; padding:22px 0 0; }
    @media(max-width:700px) { .hero { padding:28px 22px; } .hero-title { font-size:38px; } }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    foods = load_foods()
    model_foods, scaler, knn = build_knn_model(foods)
except Exception as exc:
    st.error(f"Could not load the food dataset: {exc}")
    st.info("Check that data/indb_foods.csv is present beside app.py.")
    st.stop()

# Brand header and compact top navigation
st.markdown(
    """
    <div class="brandline">
      <div class="brandmark">✳</div>
      <div><div class="brandname">nutriplan<span style="color:#75946f">.</span>ai</div>
      <div class="brandtag">Indian food • smarter choices</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)
page = st.radio(
    "Main navigation",
    ["Overview", "Build my plan", "Food library", "Dataset & insights"],
    horizontal=True,
    label_visibility="collapsed",
)

if page == "Overview":
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-kicker">✳ YOUR EVERYDAY FOOD COMPANION</div>
          <div class="hero-title">Eat well.<br><span>Live in balance.</span></div>
          <div class="hero-copy">A simple, data-powered way to explore Indian foods and discover options that fit your nutrition goals — built on real food composition data.</div>
          <div class="hero-pill">✦ Powered by 1,014 foods from the Indian Nutrient Databank</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_tile("Foods in dataset", f"{len(foods):,}", "Indian food entries")
    with c2:
        metric_tile("Nutrients used", "6", "Energy, macros, sugar & fibre")
    with c3:
        metric_tile("Recommendation model", "KNN", "Nearest nutrient profiles")
    st.write("")
    st.markdown('<div class="section-kicker">A SIMPLE WAY TO GET STARTED</div><div class="section-title">Your food plan, in three steps</div><div class="section-sub">No complicated tracking. Just useful food ideas backed by a real dataset.</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    steps = [
        ("01", "Tell us your goal", "Add a few basic details and choose your nutrition goal."),
        ("02", "Explore matched foods", "KNN searches the databank for foods with similar nutrient profiles."),
        ("03", "Make informed choices", "Compare calories, protein, carbs, fat and fibre per 100 g."),
    ]
    for col, (num, title, body) in zip(cols, steps):
        with col:
            st.markdown(
                f'<div class="food-card" style="min-height:145px"><div class="food-rank">{num}</div><div class="food-title" style="margin-top:14px">{title}</div><div style="color:#829087;font-size:12px;line-height:1.55;margin-top:7px">{body}</div></div>',
                unsafe_allow_html=True,
            )
    st.markdown('<div class="info-strip">Dataset values are shown per 100 g so different foods can be compared consistently. Portion sizes and recipes can vary. NutriPlan AI is an educational college prototype, not a medical diet prescription.</div>', unsafe_allow_html=True)

elif page == "Build my plan":
    st.markdown('<div class="section-kicker">PERSONAL FOOD DISCOVERY</div><div class="section-title">Build your nutrition plan</div><div class="section-sub">Add a few details to estimate a daily energy range and find foods with a matching nutrient profile.</div>', unsafe_allow_html=True)
    with st.form("profile_form"):
        left, right = st.columns(2)
        with left:
            age = st.number_input("Age", min_value=18, max_value=90, value=24, step=1)
            sex = st.selectbox("Sex used for calorie formula", ["Prefer not to say", "Male", "Female"])
            height = st.number_input("Height (cm)", min_value=130.0, max_value=220.0, value=166.0, step=1.0)
            weight = st.number_input("Weight (kg)", min_value=35.0, max_value=250.0, value=70.0, step=0.5)
        with right:
            activity = st.selectbox(
                "Activity level",
                ["Mostly sitting", "Light activity (1–3 days/week)", "Moderate activity (3–5 days/week)", "Very active (6–7 days/week)"],
            )
            goal = st.selectbox("Your goal", ["Lose weight gently", "Maintain weight", "Build muscle / gain"])
            avoid_words = st.text_input("Foods to avoid by name (optional)", placeholder="e.g. peanut, fish, egg")
            st.markdown('<div class="info-strip">The avoid field checks food names only. The dataset does not provide complete ingredient or allergen lists.</div>', unsafe_allow_html=True)
        submitted = st.form_submit_button("Create my food plan  →", use_container_width=True)

    if submitted:
        st.session_state["nutrition_plan"] = {
            "age": int(age), "sex": sex, "height": float(height), "weight": float(weight),
            "activity": activity, "goal": goal, "avoid_words": avoid_words,
        }

    if "nutrition_plan" in st.session_state:
        profile = st.session_state["nutrition_plan"]
        needs = estimate_daily_needs(
            profile["age"], profile["sex"], profile["height"], profile["weight"],
            profile["activity"], profile["goal"],
        )
        st.write("")
        st.markdown('<div class="section-kicker">YOUR ESTIMATED DAILY GUIDE</div><div class="section-title">A starting point, not a strict rule</div>', unsafe_allow_html=True)
        a, b, c, d = st.columns(4)
        with a: metric_tile("Daily energy", f'{needs["calories"]:,.0f} kcal', "Estimated goal target")
        with b: metric_tile("Protein", f'{needs["protein"]:.0f} g', "Illustrative daily split")
        with c: metric_tile("Carbs", f'{needs["carbs"]:.0f} g', "Illustrative daily split")
        with d: metric_tile("Fat", f'{needs["fat"]:.0f} g', "Illustrative daily split")
        st.markdown('<div class="info-strip">Energy is estimated with the Mifflin–St Jeor equation and an activity multiplier. Macro values use a simple 25% protein / 45% carbohydrate / 30% fat split for demonstration. Individual needs vary; a qualified dietitian can tailor these targets.</div>', unsafe_allow_html=True)
        st.write("")
        st.markdown('<div class="section-kicker">KNN FOOD MATCHES</div><div class="section-title">Food ideas for your day</div><div class="section-sub">The model finds the closest foods to a goal-specific nutrient profile. Values on each card are per 100 g, not a suggested serving.</div>', unsafe_allow_html=True)

        meal_info = [
            ("Breakfast", "A balanced start to the day"),
            ("Lunch", "Hearty options for your main meal"),
            ("Snack", "Lighter options between meals"),
            ("Dinner", "Nutrient-profile matches for the evening"),
        ]
        already_used = set()
        for meal, desc in meal_info:
            st.markdown(f'<div style="font-family:Manrope,sans-serif;font-size:18px;font-weight:800;color:#244e3d;margin:15px 0 3px">{meal}</div><div style="font-size:11px;color:#829087;margin-bottom:10px">{desc}</div>', unsafe_allow_html=True)
            matches = recommend_foods(
                model_foods, scaler, knn, profile["goal"], meal,
                profile["avoid_words"], already_used, count=3,
            )
            if not matches:
                st.warning(f"No {meal.lower()} matches found. Try removing some avoid keywords.")
                continue
            cards = st.columns(3)
            for rank, ((row, distance), col) in enumerate(zip(matches, cards), start=1):
                with col:
                    food_card(row, rank, distance)
                already_used.add(str(row.get("food_code", "")))
        st.caption("Recommendations are nearest-neighbour matches from the uploaded INDB food records. They are not a validated medical diet plan.")

else:
    pass

if page == "Food library":
    st.markdown('<div class="section-kicker">EXPLORE THE DATABANK</div><div class="section-title">Indian food library</div><div class="section-sub">Search foods and compare their nutrient values on a per-100-g basis.</div>', unsafe_allow_html=True)
    search_col, source_col = st.columns([2, 1])
    with search_col:
        query = st.text_input("Search foods", placeholder="Search idli, dal, rice, paneer...")
    with source_col:
        source_values = ["All sources"] + sorted(foods["primary_source"].dropna().unique().tolist())
        source_filter = st.selectbox("Data source", source_values)
    f1, f2, f3 = st.columns(3)
    with f1:
        min_protein = st.slider("Minimum protein (g / 100 g)", 0.0, 30.0, 0.0, 1.0)
    with f2:
        max_calories = st.slider("Maximum calories (kcal / 100 g)", 50, 650, 650, 25)
    with f3:
        sort_by = st.selectbox("Sort by", ["Protein (high to low)", "Calories (low to high)", "Fibre (high to low)", "Name"])
    filtered = foods.copy()
    if query:
        filtered = filtered[filtered["food_name"].str.contains(query, case=False, na=False)]
    if source_filter != "All sources":
        filtered = filtered[filtered["primary_source"] == source_filter]
    filtered = filtered[
        (filtered["protein_g_100g"].fillna(0) >= min_protein)
        & (filtered["energy_kcal_100g"].fillna(99999) <= max_calories)
    ]
    sort_map = {
        "Protein (high to low)": ("protein_g_100g", False),
        "Calories (low to high)": ("energy_kcal_100g", True),
        "Fibre (high to low)": ("fibre_g_100g", False),
        "Name": ("food_name", True),
    }
    col, asc = sort_map[sort_by]
    filtered = filtered.sort_values(col, ascending=asc)
    st.markdown(f'<div style="color:#809087;font-size:12px;margin:9px 0 10px">Showing <b style="color:#244e3d">{len(filtered):,}</b> of {len(foods):,} food entries</div>', unsafe_allow_html=True)
    display_cols = [
        "food_name", "energy_kcal_100g", "protein_g_100g", "carbohydrate_g_100g",
        "fat_g_100g", "free_sugar_g_100g", "fibre_g_100g", "primary_source",
    ]
    view = filtered[display_cols].rename(columns={
        "food_name": "Food",
        "energy_kcal_100g": "kcal / 100g",
        "protein_g_100g": "Protein (g)",
        "carbohydrate_g_100g": "Carbs (g)",
        "fat_g_100g": "Fat (g)",
        "free_sugar_g_100g": "Free sugar (g)",
        "fibre_g_100g": "Fibre (g)",
        "primary_source": "Data source",
    })
    st.dataframe(view.head(100), use_container_width=True, hide_index=True)
    st.caption("Showing up to 100 matches. All nutrient values are per 100 g.")

elif page == "Dataset & insights":
    st.markdown('<div class="section-kicker">TRANSPARENT DATA</div><div class="section-title">Dataset & insights</div><div class="section-sub">Real records from the Anuvaad Indian Nutrient Databank workbook you provided.</div>', unsafe_allow_html=True)
    source_counts = foods["primary_source"].value_counts().rename_axis("Source").reset_index(name="Food records")
    total_rows = len(foods)
    missing_critical = int(foods[["food_name"] + FEATURES].isna().sum().sum())
    m1, m2, m3, m4 = st.columns(4)
    with m1: metric_tile("Food records", f"{total_rows:,}", "Rows excluding header")
    with m2: metric_tile("Original columns", "82", "Fields in source workbook")
    with m3: metric_tile("Fields used here", "16", "Nutrition + food metadata")
    with m4: metric_tile("Missing core values", f"{missing_critical:,}", "Name and six nutrients")
    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="section-title" style="font-size:18px">Records by source</div>', unsafe_allow_html=True)
        fig = px.bar(
            source_counts, x="Source", y="Food records",
            color="Food records", color_continuous_scale=["#d9e9c7", "#315f47"],
        )
        fig.update_layout(
            margin=dict(l=5, r=5, t=10, b=5), height=300,
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(color="#52695b"), coloraxis_showscale=False,
        )
        fig.update_xaxes(title="", tickangle=-15, showgrid=False)
        fig.update_yaxes(title="Food records", gridcolor="#edf1ea")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown('<div class="section-title" style="font-size:18px">Calories per 100 g</div>', unsafe_allow_html=True)
        fig2 = px.histogram(
            foods.dropna(subset=["energy_kcal_100g"]),
            x="energy_kcal_100g", nbins=28,
            color_discrete_sequence=["#83a77c"],
        )
        fig2.update_layout(
            margin=dict(l=5, r=5, t=10, b=5), height=300,
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(color="#52695b"), showlegend=False,
        )
        fig2.update_xaxes(title="Energy (kcal / 100 g)", showgrid=False)
        fig2.update_yaxes(title="Number of foods", gridcolor="#edf1ea")
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown('<div class="section-title" style="font-size:18px">Dataset preview</div>', unsafe_allow_html=True)
    preview_cols = [
        "food_code", "food_name", "primary_source", "energy_kcal_100g",
        "protein_g_100g", "carbohydrate_g_100g", "fat_g_100g", "fibre_g_100g",
    ]
    st.dataframe(foods[preview_cols].head(12), use_container_width=True, hide_index=True)
    st.markdown('<div class="info-strip">Source workbook: Anuvaad_INDB_2024.11.xlsx. The original sheet contains 1,014 food records and 82 columns. This prototype uses a selected 16-field extract. Some per-serving values are missing, so the app consistently compares nutrients per 100 g.</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer-note">NutriPlan AI · Academic AI/ML prototype · Nutrient figures come from the included INDB dataset. Not medical advice.</div>',
    unsafe_allow_html=True,
)
