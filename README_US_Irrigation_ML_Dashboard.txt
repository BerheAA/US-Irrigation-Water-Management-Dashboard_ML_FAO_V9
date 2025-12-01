
US Irrigation Water Management Dashboard (FAO + ML Yield)
=========================================================

This app is a clean, corrected version of the earlier
`us-irrigation-water-management-dashboard-train-ml_v4` concept, with two key fixes:

1. Weather & ET summary:
   - Uses robust column checks (no KeyError).
   - Summaries and plots are based on `date`, `ETc_mm`, `ETa_mm`, `rain_mm`,
     and `irr_gross_mm` from the FAO soil-water balance engine.

2. ML yield estimation that *responds to irrigation*:
   - You upload a CSV with yield and water indicators
     (e.g., APSIM/DSSAT/AquaCrop outputs or field trials).
   - A RandomForestRegressor is trained on seasonal totals:
       * total irrigation (mm)
       * seasonal rainfall (mm)
       * seasonal ETc (mm)
       * optionally seasonal ETa (mm)
   - When you run a FAO irrigation scenario, the engine computes the same
     features for that scenario (totals of irrigation, rainfall, ETc, ETa),
     and **those values feed directly into the ML model**.
   - As a result, changing irrigation system, MAD, efficiency, or climate
     source *changes the features*, and the ML-predicted yield responds
     accordingly—no more flat, non-responsive yield curve.

Key features
------------
- FAO-style soil water balance in **mm** (ETc, ETa, TAW, RAW).
- Fixed **25 mm gross** per irrigation event for Center pivot and Drip/Micro
  (≈1 inch), with the number of events controlled by MAD.
- Surface/Furrow refills to field capacity in a single event.
- Weather and ET can come from:
  - Open-Meteo ERA5 (automatic),
  - a synthetic ETo pattern (demo),
  - or uploaded daily CSV (ET0 + rainfall).
- Sections:
  1. Field location (map)
  2. Climate / ET source
  3. Weather & ET summary (plots)
  4. Scenario results: irrigation & ET
  5. Detailed irrigation schedule
  6. Train ML yield model from your data
  7. ML yield estimate for the current irrigation scenario, including a simple
     yield-vs-irrigation sensitivity curve.

How to run
----------
1. Install dependencies (in your environment):
   - streamlit
   - pandas
   - numpy
   - requests
   - scikit-learn

2. From the folder containing `us_irrigation_ml_dashboard_app.py`, run:

   streamlit run us_irrigation_ml_dashboard_app.py

3. In the app:
   - Use the sidebar to configure location, crop, soil, system, and MAD.
   - Choose a climate source (automatic, synthetic, or CSV).
   - Click **Generate irrigation scenario**.
   - Upload a yield dataset in Section 6 and train an ML model.
   - The model uses water variables; Section 7 then gives a scenario-specific
     ML yield prediction that responds to irrigation level and ET conditions.

To patch your existing Streamlit Cloud app, you can:
- Either replace its `app.py` with this script, or
- Copy the ML-related sections (6–7) and FAO feature mapping logic into your
  current app around your own layout.

This version is designed to avoid the KeyError and ensure ML yields are
scenario-responsive, addressing the issues you described.
