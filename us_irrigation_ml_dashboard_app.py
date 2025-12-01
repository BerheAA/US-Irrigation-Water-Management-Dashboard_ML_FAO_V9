
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ---------------------------------------------------
# CONFIG & THEME
# ---------------------------------------------------
st.set_page_config(
    page_title="US Irrigation Water Management Dashboard (FAO + ML Yield)",
    layout="wide"
)

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background: #f5f7fb;
    }
    [data-testid="stSidebar"] {
        background-color: #eef1f7;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 650;
        color: #1f4e79;
        margin-top: 0.4rem;
        margin-bottom: 0.2rem;
    }
    .card {
        padding: 0.6rem 0.8rem;
        border-radius: 0.6rem;
        background-color: #ffffff;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.12);
        margin-bottom: 0.7rem;
    }
    .card-blue { border-left: 4px solid #1f4e79; }
    .card-green { border-left: 4px solid #2e7d32; }
    .card-orange { border-left: 4px solid #f57c00; }
    .tiny-text { font-size: 0.8rem; }
    </style>
    """, unsafe_allow_html=True,
)

st.title("US Irrigation Water Management Dashboard (FAO + ML Yield)")

# ---------------------------------------------------
# STATE CENTROIDS (APPROX)
# ---------------------------------------------------
STATE_CENTROIDS = {
    "Alabama": (32.8, -86.8),
    "Alaska": (64.2, -149.5),
    "Arizona": (34.0, -112.1),
    "Arkansas": (34.8, -92.2),
    "California": (37.2, -119.7),
    "Colorado": (39.0, -105.5),
    "Connecticut": (41.6, -72.7),
    "Delaware": (39.0, -75.5),
    "Florida": (28.6, -82.4),
    "Georgia": (32.7, -83.4),
    "Hawaii": (20.8, -156.3),
    "Idaho": (44.2, -114.1),
    "Illinois": (40.3, -89.0),
    "Indiana": (40.2, -86.1),
    "Iowa": (42.0, -93.3),
    "Kansas": (38.5, -98.0),
    "Kentucky": (37.5, -85.3),
    "Louisiana": (31.0, -92.0),
    "Maine": (45.3, -69.0),
    "Maryland": (39.0, -76.7),
    "Massachusetts": (42.3, -71.8),
    "Michigan": (44.3, -85.4),
    "Minnesota": (46.3, -94.0),
    "Mississippi": (32.7, -89.7),
    "Missouri": (38.5, -92.5),
    "Montana": (46.9, -110.4),
    "Nebraska": (41.5, -99.8),
    "Nevada": (39.5, -116.9),
    "New Hampshire": (43.7, -71.6),
    "New Jersey": (40.1, -74.7),
    "New Mexico": (34.2, -106.0),
    "New York": (42.9, -75.5),
    "North Carolina": (35.5, -79.4),
    "North Dakota": (47.5, -100.5),
    "Ohio": (40.4, -82.8),
    "Oklahoma": (35.6, -97.5),
    "Oregon": (44.0, -120.6),
    "Pennsylvania": (41.0, -77.8),
    "Rhode Island": (41.7, -71.6),
    "South Carolina": (33.8, -80.9),
    "South Dakota": (44.4, -100.2),
    "Tennessee": (35.8, -86.3),
    "Texas": (31.0, -99.3),
    "Utah": (39.3, -111.7),
    "Vermont": (44.1, -72.7),
    "Virginia": (37.5, -78.8),
    "Washington": (47.4, -120.7),
    "West Virginia": (38.6, -80.6),
    "Wisconsin": (44.5, -89.8),
    "Wyoming": (43.0, -107.6),
}
state_list = sorted(list(STATE_CENTROIDS.keys()))

# ---------------------------------------------------
# CROP PARAMETERS (Kc curves)
# ---------------------------------------------------
CROP_PARAMS = {
    "Corn (grain)": {
        "kc_init": 0.4,
        "kc_mid": 1.20,
        "kc_end": 0.65,
    },
    "Winter wheat": {
        "kc_init": 0.35,
        "kc_mid": 1.15,
        "kc_end": 0.25,
    },
    "Grain sorghum": {
        "kc_init": 0.35,
        "kc_mid": 1.05,
        "kc_end": 0.45,
    },
    "Alfalfa (hay)": {
        "kc_init": 0.90,
        "kc_mid": 1.20,
        "kc_end": 1.05,
    },
    "Cotton": {
        "kc_init": 0.35,
        "kc_mid": 1.15,
        "kc_end": 0.60,
    },
    "Pasture / Grass": {
        "kc_init": 0.70,
        "kc_mid": 1.00,
        "kc_end": 0.90,
    },
}

# ---------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------
def fetch_open_meteo_daily(lat, lon, start_date, end_date):
    """Fetch daily ET0 (FAO) and precipitation from Open-Meteo ERA5."""
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "et0_fao_evapotranspiration,precipitation_sum",
        "timezone": "America/Chicago",
    }
    try:
        resp = requests.get(base_url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})
        times = daily.get("time", [])
        et0_mm = daily.get("et0_fao_evapotranspiration", [])
        rain_mm = daily.get("precipitation_sum", [])

        if not (times and et0_mm and rain_mm):
            raise ValueError("Missing ET0 or precipitation in Open-Meteo response.")

        df = pd.DataFrame({
            "date": pd.to_datetime(times),
            "ET0_mm": et0_mm,
            "rain_mm": rain_mm
        })
        return df
    except Exception as e:
        st.warning(f"Open-Meteo data retrieval failed ({e}). Using synthetic ETo pattern instead.")
        return None


def generate_synthetic_weather(start_date, season_length):
    np.random.seed(42)
    dates = [start_date + timedelta(days=i) for i in range(season_length)]
    eto = np.random.normal(loc=4.5, scale=0.8, size=season_length)  # mm/day
    eto = np.clip(eto, 3.0, 7.0)
    rain = np.zeros(season_length)
    storm_days = np.random.choice(range(season_length), size=season_length // 6, replace=False)
    for d in storm_days:
        rain[d] = np.random.uniform(5.0, 30.0)
    return pd.DataFrame({"date": dates, "ET0_mm": eto, "rain_mm": rain})


def run_soil_water_balance_mm(weather, awc_mm_per_m, root_depth_m, mad_fraction, app_eff, crop_type, system_type):
    """FAO-style soil water balance in mm with simple irrigation rules.
    - Center pivot / Drip-Micro: fixed 25 mm gross per event (~1 inch)
    - Surface: refill to field capacity
    Returns daily dataframe + seasonal totals.
    """
    weather = weather.copy()
    season_length = len(weather)

    defaults = CROP_PARAMS.get(crop_type, CROP_PARAMS["Corn (grain)"])
    kc_init = defaults["kc_init"]
    kc_mid = defaults["kc_mid"]
    kc_end = defaults["kc_end"]

    d = np.arange(season_length)
    kc = np.piecewise(
        d,
        [d < season_length * 0.2,
         (d >= season_length * 0.2) & (d < season_length * 0.7),
         d >= season_length * 0.7],
        [kc_init, kc_mid, kc_end]
    )
    weather["Kc"] = kc
    weather["ETc_mm"] = weather["ET0_mm"] * weather["Kc"]

    TAW = awc_mm_per_m * root_depth_m
    RAW = mad_fraction * TAW

    soil_storage = []
    deficit = []
    irr_net = []
    irr_gross = []

    sw = TAW
    for _, row in weather.iterrows():
        sw = sw + row["rain_mm"] - row["ETc_mm"]
        if sw > TAW:
            sw = TAW
        if sw < 0:
            sw = 0.0

        dep = TAW - sw
        net = 0.0
        gross = 0.0

        if dep > RAW:
            if system_type in ["Center pivot", "Drip / Micro"]:
                gross = 25.0  # mm
                net = gross * app_eff
                sw = min(sw + net, TAW)
            else:
                net = dep
                gross = net / app_eff
                sw = TAW

        soil_storage.append(sw)
        deficit.append(TAW - sw)
        irr_net.append(net)
        irr_gross.append(gross)

    weather["soil_storage_mm"] = soil_storage
    weather["deficit_mm"] = deficit
    weather["irr_net_mm"] = irr_net
    weather["irr_gross_mm"] = irr_gross

    dep_arr = weather["deficit_mm"].values
    ks = np.ones_like(dep_arr, dtype=float)
    mask_stress = dep_arr > RAW
    if TAW > RAW:
        ks[mask_stress] = np.maximum(
            0.0,
            (TAW - dep_arr[mask_stress]) / (TAW - RAW)
        )
    weather["Ks"] = ks
    weather["ETa_mm"] = weather["ETc_mm"] * weather["Ks"]

    totals = {
        "TAW": TAW,
        "RAW": RAW,
        "total_irr_gross_mm": weather["irr_gross_mm"].sum(),
        "total_irr_net_mm": weather["irr_net_mm"].sum(),
        "n_events": int((weather["irr_gross_mm"] > 0).sum()),
        "ETc_season_mm": weather["ETc_mm"].sum(),
        "ETa_season_mm": weather["ETa_mm"].sum(),
        "total_rain_mm": weather["rain_mm"].sum(),
    }
    return weather, totals


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.header("Location & Season")
state = st.sidebar.selectbox("State", state_list, index=state_list.index("Kansas"))
default_lat, default_lon = STATE_CENTROIDS[state]

lat = st.sidebar.number_input(
    "Latitude (°N)",
    min_value=-90.0, max_value=90.0,
    value=float(default_lat), step=0.01
)
lon = st.sidebar.number_input(
    "Longitude (°E)",
    min_value=-180.0, max_value=180.0,
    value=float(default_lon), step=0.01
)

season_length = st.sidebar.number_input(
    "Season length (days)",
    min_value=60, max_value=220,
    value=120, step=5
)
start_date = st.sidebar.date_input(
    "Planting / emergence date",
    value=date(date.today().year, 5, 1)
)
end_date = start_date + timedelta(days=int(season_length) - 1)
st.sidebar.caption(f"Simulation period: {start_date} → {end_date}")

st.sidebar.header("Crop & Soil")
crop_type = st.sidebar.selectbox(
    "Crop type",
    list(CROP_PARAMS.keys()),
    index=0
)

soil_type = st.sidebar.selectbox(
    "Soil textural group",
    ["Sandy loam", "Loam", "Silt loam", "Clay loam"],
    index=2
)
if soil_type == "Sandy loam":
    default_awc = 120.0
elif soil_type == "Loam":
    default_awc = 160.0
elif soil_type == "Silt loam":
    default_awc = 200.0
else:
    default_awc = 180.0

awc_mm_per_m = st.sidebar.slider(
    "Available water (mm/m)",
    min_value=80.0, max_value=260.0,
    value=float(default_awc), step=10.0
)
root_depth_m = st.sidebar.slider(
    "Effective rooting depth (m)",
    min_value=0.5, max_value=2.5,
    value=1.5, step=0.1
)

st.sidebar.header("Irrigation System & MAD")
system_type = st.sidebar.selectbox(
    "Irrigation system",
    ["Center pivot", "Surface / Furrow", "Drip / Micro"],
    index=0
)
if system_type == "Center pivot":
    default_eff = 0.8
elif system_type == "Surface / Furrow":
    default_eff = 0.6
else:
    default_eff = 0.9

application_eff = st.sidebar.slider(
    "Application efficiency (%)",
    min_value=50, max_value=95,
    value=int(default_eff * 100), step=5
) / 100.0

mad_fraction = st.sidebar.slider(
    "Refill trigger (MAD, fraction of TAW)",
    min_value=0.3, max_value=0.7,
    value=0.5, step=0.05
)

# ---------------------------------------------------
# MAIN LAYOUT – MAP + CLIMATE SOURCE
# ---------------------------------------------------
df_weather = None
wb_weather = None
totals = None

top_col1, top_col2 = st.columns([2.0, 1.3])

with top_col1:
    st.markdown('<div class="section-title">1. Field location</div>', unsafe_allow_html=True)
    st.markdown('<div class="card card-blue">', unsafe_allow_html=True)
    st.caption("Blue point shows the approximate field location.")
    map_df = pd.DataFrame({"lat": [lat], "lon": [lon]})
    st.map(map_df, zoom=6)
    st.markdown('</div>', unsafe_allow_html=True)

with top_col2:
    st.markdown('<div class="section-title">2. Climate / ET source</div>', unsafe_allow_html=True)
    st.markdown('<div class="card card-blue tiny-text">', unsafe_allow_html=True)
    climate_source = st.radio(
        "Climate data source",
        [
            "Automatic (Open-Meteo ERA5)",
            "Use synthetic ETo pattern (demo)",
            "Upload daily climate CSV"
        ],
        index=0
    )
    st.caption("CSV option expects daily ET0 and rainfall at the field; you can map columns.")

    uploaded_df = None
    if climate_source == "Upload daily climate CSV":
        uploaded = st.file_uploader("Upload daily climate CSV", type=["csv"])
        if uploaded is not None:
            try:
                tmp = pd.read_csv(uploaded)
                st.success("CSV loaded. Map your columns below.")
                date_col = st.selectbox("Date column", tmp.columns, index=0)
                eto_col = st.selectbox(
                    "ET0 column (mm/day)",
                    [c for c in tmp.columns if c != date_col],
                    index=1 if len(tmp.columns) > 1 else 0
                )
                rain_col = st.selectbox(
                    "Rain column (mm/day)",
                    [c for c in tmp.columns if c not in [date_col, eto_col]],
                    index=0
                )
                tmp["date"] = pd.to_datetime(tmp[date_col])
                df_csv = tmp[["date", eto_col, rain_col]].rename(
                    columns={eto_col: "ET0_mm", rain_col: "rain_mm"}
                )
                df_csv = df_csv.sort_values("date")
                mask = (df_csv["date"] >= pd.to_datetime(start_date)) & (
                    df_csv["date"] <= pd.to_datetime(end_date)
                )
                df_csv = df_csv.loc[mask]
                if df_csv.empty:
                    st.warning("No rows in CSV fall within the simulation period.")
                else:
                    uploaded_df = df_csv.reset_index(drop=True)
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card tiny-text">', unsafe_allow_html=True)
    run_button = st.button("Generate irrigation scenario", type="primary")
    if not run_button:
        st.info("Set inputs, then click **Generate irrigation scenario** to run FAO soil water balance.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# RUN FAO WATER BALANCE
# ---------------------------------------------------
if run_button:
    if climate_source == "Automatic (Open-Meteo ERA5)":
        df_weather = fetch_open_meteo_daily(lat, lon, start_date, end_date)
        if df_weather is None:
            df_weather = generate_synthetic_weather(start_date, season_length)
    elif climate_source == "Use synthetic ETo pattern (demo)":
        df_weather = generate_synthetic_weather(start_date, season_length)
    else:
        if uploaded_df is not None and not uploaded_df.empty:
            df_weather = uploaded_df.copy()
        else:
            st.info("No valid CSV uploaded; using synthetic ETo pattern instead.")
            df_weather = generate_synthetic_weather(start_date, season_length)

    df_weather = df_weather.sort_values("date")
    if len(df_weather) > int(season_length):
        df_weather = df_weather.iloc[: int(season_length)]

    wb_weather, totals = run_soil_water_balance_mm(
        df_weather,
        awc_mm_per_m=awc_mm_per_m,
        root_depth_m=root_depth_m,
        mad_fraction=mad_fraction,
        app_eff=application_eff,
        crop_type=crop_type,
        system_type=system_type,
    )

# ---------------------------------------------------
# 3. WEATHER & ET SUMMARY
# ---------------------------------------------------
st.markdown('<div class="section-title">3. Weather & ET summary</div>', unsafe_allow_html=True)
if wb_weather is None:
    st.info("Generate a scenario first to summarize weather and ET.")
else:
    with st.expander("Daily ETc, ETa, rainfall, irrigation (mm)", expanded=True):
        cols = list(wb_weather.columns)
        if "date" in cols and "ETc_mm" in cols and "ETa_mm" in cols and "rain_mm" in cols and "irr_gross_mm" in cols:
            ts = wb_weather[["date", "ETc_mm", "ETa_mm", "rain_mm", "irr_gross_mm"]].copy()
            ts = ts.set_index("date")
            st.line_chart(ts[["ETc_mm", "ETa_mm"]])
            st.bar_chart(ts[["rain_mm"]])
            st.bar_chart(ts[["irr_gross_mm"]])
        else:
            st.warning("Expected ET or irrigation columns are missing in this run.")

# ---------------------------------------------------
# 4. SCENARIO RESULTS: IRRIGATION & ET
# ---------------------------------------------------
st.markdown('<div class="section-title">4. Scenario results: irrigation & ET</div>', unsafe_allow_html=True)
if wb_weather is None or totals is None:
    st.info("Generate a scenario first to see seasonal irrigation and ET totals.")
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card card-orange tiny-text">', unsafe_allow_html=True)
        st.markdown(f"**Irrigation – {crop_type}**")
        st.metric("Total gross irrigation (mm)", f"{totals['total_irr_gross_mm']:.0f}")
        st.metric("Total net irrigation (mm)", f"{totals['total_irr_net_mm']:.0f}")
        st.metric("Number of irrigation events", int(totals["n_events"]))
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card card-green tiny-text">', unsafe_allow_html=True)
        st.markdown("**Water balance**")
        st.metric("Seasonal ETc (mm)", f"{totals['ETc_season_mm']:.0f}")
        st.metric("Seasonal ETa (mm)", f"{totals['ETa_season_mm']:.0f}")
        st.metric("Seasonal rainfall (mm)", f"{totals['total_rain_mm']:.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card card-blue tiny-text">', unsafe_allow_html=True)
        st.markdown("**Soil storage capacity**")
        st.metric("TAW (mm)", f"{totals['TAW']:.0f}")
        st.metric("RAW (mm)", f"{totals['RAW']:.0f}")
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# 5. IRRIGATION SCHEDULE TABLE
# ---------------------------------------------------
st.markdown('<div class="section-title">5. Detailed irrigation schedule</div>', unsafe_allow_html=True)
if wb_weather is None:
    st.info("Generate a scenario first to compute the irrigation schedule.")
else:
    events = wb_weather[wb_weather["irr_gross_mm"] > 0][["date", "irr_net_mm", "irr_gross_mm"]]
    events = events.rename(
        columns={
            "date": "Date",
            "irr_net_mm": "Net irrigation (mm)",
            "irr_gross_mm": "Gross irrigation (mm)",
        }
    )
    if events.empty:
        st.info("No irrigation events were triggered for this configuration.")
    else:
        st.dataframe(events, height=260)

# ---------------------------------------------------
# 6. TRAIN ML YIELD MODEL FROM YOUR DATA
# ---------------------------------------------------
st.markdown('<div class="section-title">6. Train ML yield model from your data</div>', unsafe_allow_html=True)

if "ml_model" not in st.session_state:
    st.session_state["ml_model"] = None
if "ml_features" not in st.session_state:
    st.session_state["ml_features"] = None
if "ml_target_name" not in st.session_state:
    st.session_state["ml_target_name"] = None

st.markdown('<div class="card card-blue tiny-text">', unsafe_allow_html=True)
st.markdown("**Upload trial dataset (APSIM/DSSAT/AquaCrop or field trials)**", unsafe_allow_html=True)

ml_file = st.file_uploader("Upload CSV with yield and water indicators", type=["csv"], key="ml_file")
if ml_file is not None:
    try:
        df_ml = pd.read_csv(ml_file)
        st.success("ML data loaded. Map columns below.")
        st.write("Preview:", df_ml.head())

        target_col = st.selectbox("Yield column (target)", df_ml.columns)
        irr_col = st.selectbox("Total irrigation column (mm)", df_ml.columns)
        rain_col_ml = st.selectbox("Total rainfall column (mm)", df_ml.columns)
        etc_col = st.selectbox("Seasonal ETc column (mm)", df_ml.columns)
        eta_col = st.selectbox("Seasonal ETa column (mm, optional)", ["<not used>"] + list(df_ml.columns))

        feature_cols = [irr_col, rain_col_ml, etc_col]
        if eta_col != "<not used>":
            feature_cols.append(eta_col)

        X = df_ml[feature_cols].astype(float)
        y = df_ml[target_col].astype(float)

        model = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            max_depth=None,
            min_samples_leaf=3,
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        st.session_state["ml_model"] = model
        st.session_state["ml_features"] = feature_cols
        st.session_state["ml_target_name"] = target_col

        st.success(f"ML model trained. In-sample R² = {r2:.2f}")
        st.caption(
            "This model explicitly uses seasonal irrigation, rainfall, ETc, and optionally ETa. "
            "Therefore, when you change irrigation strategy in the FAO engine, the predicted yield will respond."
        )
    except Exception as e:
        st.error(f"Error training ML model: {e}")
else:
    st.caption("Upload a CSV to train an ML yield model that responds to irrigation and ET conditions.")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# 7. ML YIELD ESTIMATION FOR CURRENT SCENARIO
# ---------------------------------------------------
st.markdown('<div class="section-title">7. ML yield estimate for this irrigation scenario</div>', unsafe_allow_html=True)

ml_model = st.session_state.get("ml_model", None)
feature_cols = st.session_state.get("ml_features", None)
target_name = st.session_state.get("ml_target_name", None)

if wb_weather is None or totals is None:
    st.info("Generate an irrigation scenario first (Sections 1–5).")
elif ml_model is None or feature_cols is None:
    st.info("Train an ML yield model in Section 6 to enable scenario-based yield predictions.")
else:
    # Build feature vector from CURRENT scenario so yield responds to irrigation
    feat_values = {}
    if "total_irr_gross_mm" in totals:
        feat_values["total_irr_gross_mm"] = totals["total_irr_gross_mm"]
    if "total_rain_mm" in totals:
        feat_values["total_rain_mm"] = totals["total_rain_mm"]
    if "ETc_season_mm" in totals:
        feat_values["ETc_season_mm"] = totals["ETc_season_mm"]
    if "ETa_season_mm" in totals:
        feat_values["ETa_season_mm"] = totals["ETa_season_mm"]

    # Map these to user-selected ML feature columns:
    #   any feature name containing "irr" gets total_irr_gross_mm
    #   any containing "rain" gets total_rain_mm
    #   any containing "ETc" or "etc" gets ETc_season_mm
    #   any containing "ETa" or "eta" gets ETa_season_mm
    X_scenario = []
    missing_any = False
    for col in feature_cols:
        cl = col.lower()
        if "irr" in cl:
            val = totals["total_irr_gross_mm"]
        elif "rain" in cl or "precip" in cl:
            val = totals["total_rain_mm"]
        elif "etc" in cl:
            val = totals["ETc_season_mm"]
        elif "eta" in cl:
            val = totals["ETa_season_mm"]
        else:
            val = 0.0
            missing_any = True
        X_scenario.append(val)
    X_scenario = np.array(X_scenario).reshape(1, -1)

    y_ml = ml_model.predict(X_scenario)[0]

    st.markdown('<div class="card card-orange tiny-text">', unsafe_allow_html=True)
    st.markdown(f"**ML-predicted yield for this scenario**")
    st.markdown(f"Predicted {target_name}: **{y_ml:.1f}**")
    st.caption(
        "This prediction explicitly depends on seasonal irrigation, rainfall, ETc, and optionally ETa from the current FAO scenario. "
        "If you change irrigation system, MAD, efficiency, or climate source, the features are recomputed and the ML yield updates."
    )
    if missing_any:
        st.caption(
            "One or more ML feature columns could not be automatically mapped from the scenario; "
            "those were set to 0. To avoid this, use column names containing 'irr', 'rain', 'ETc', or 'ETa' "
            "for the respective water variables when training the ML model."
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Simple sensitivity curve of yield vs irrigation (scaling irrigation)
    with st.expander("Yield response curve vs irrigation (ML-based)", expanded=False):
        irr_scale = np.linspace(0.5, 1.5, 9)
        irr_vals = []
        y_vals = []
        for s_factor in irr_scale:
            X_s = []
            for col in feature_cols:
                cl = col.lower()
                if "irr" in cl:
                    val = totals["total_irr_gross_mm"] * s_factor
                elif "rain" in cl or "precip" in cl:
                    val = totals["total_rain_mm"]
                elif "etc" in cl:
                    val = totals["ETc_season_mm"]
                elif "eta" in cl:
                    val = totals["ETa_season_mm"]
                else:
                    val = 0.0
                X_s.append(val)
            X_s = np.array(X_s).reshape(1, -1)
            y_hat = ml_model.predict(X_s)[0]
            irr_vals.append(totals["total_irr_gross_mm"] * s_factor)
            y_vals.append(y_hat)

        df_sens = pd.DataFrame({"Total irrigation (mm)": irr_vals, f"{target_name} (ML)": y_vals})
        df_sens = df_sens.set_index("Total irrigation (mm)")
        st.line_chart(df_sens)

st.markdown(
    """
    <div class="tiny-text">
    ML yield estimation in this dashboard is **explicitly driven by water variables**: seasonal irrigation, rainfall, ETc, and optionally ETa.
    The FAO engine computes those for each scenario; they are then fed into the trained ML model so the predicted yield responds to irrigation
    and climate changes, avoiding the “flat” or non-responsive behavior you observed in the previous app.
    </div>
    """, unsafe_allow_html=True,
)
