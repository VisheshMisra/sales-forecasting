from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Rossmann Sales Forecast", page_icon="📊", layout="wide")

APP_DIR = Path(__file__).parent
REPORT_DIR = APP_DIR.parent / "reports"
FIG_DIR = REPORT_DIR / "figures"
DATA_PATH = APP_DIR / "app_data.csv"
MODEL_PATH = APP_DIR.parent / "models" / "lgbm_final.pkl"
FEATURES_PATH = APP_DIR / "forecast_features.parquet"

VALID_START = pd.Timestamp("2015-06-19")
FORECAST_START = pd.Timestamp("2015-08-01")
ACTUAL_COLOR = "#1F3A5F"
FORECAST_COLOR = "#F28E2B"

# ----------------------------------------------------------------------------- styling
st.markdown(
    """
    <style>
    .block-container {padding-top: 4.5rem; max-width: 1200px;}
    .hero {background: linear-gradient(120deg, #1F3A5F 0%, #2A7F8E 100%);
           padding: 1.3rem 1.6rem; border-radius: 14px; margin-bottom: 1.1rem;}
    .hero-title {color: #ffffff; font-size: 1.75rem; font-weight: 700; line-height: 1.2;}
    .hero-sub {color: #e6f0f3; font-size: 0.95rem; margin-top: 0.3rem;}
    [data-testid="stMetric"] {background: rgba(128,128,128,0.08);
           border: 1px solid rgba(128,128,128,0.25); padding: 0.8rem 1rem;
           border-radius: 12px;}
    button[data-baseweb="tab"] {font-size: 1rem; font-weight: 600;}
    footer {visibility: hidden;}
    </style>
    <div class="hero">
      <div class="hero-title">Rossmann Store Sales Forecast</div>
      <div class="hero-sub">Daily sales forecasts for 1,100+ drugstores, 1 Aug to 17 Sep 2015
      &middot; LightGBM model</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- helpers
@st.cache_data
def load_data(file_version):
    # file_version (the file's last-modified time) makes Streamlit reload the data
    # whenever app_data.csv is replaced, instead of serving a stale cached copy.
    return pd.read_csv(DATA_PATH, parse_dates=["Date"])


@st.cache_data
def load_forecast_features(file_version):
    return pd.read_parquet(FEATURES_PATH)


@st.cache_resource
def load_model(file_version):
    return joblib.load(MODEL_PATH)


def compute_mape(frame):
    """Mean absolute percentage error on days with both an actual and a forecast."""
    chk = frame.dropna(subset=["Sales", "Forecast"])
    chk = chk[chk["Sales"] > 0]
    if len(chk) == 0:
        return None
    return float(((chk["Sales"] - chk["Forecast"]).abs() / chk["Sales"]).mean() * 100)


def fmt(x):
    return "{:,.0f}".format(x)


def filter_forecast_range(frame, start_date, end_date):
    filtered = frame.copy()
    outside_range = (
        (filtered.index >= FORECAST_START)
        & ((filtered.index < start_date) | (filtered.index > end_date))
    )
    filtered.loc[outside_range, "Forecast"] = float("nan")
    return filtered


def actual_vs_forecast_chart(frame):
    """Line chart of actual vs forecast. Closed days (zero sales) are left out."""
    plot = frame.reset_index()[["Date", "Sales", "Forecast"]].copy()
    plot = plot.rename(columns={"Sales": "Actual"})
    for c in ["Actual", "Forecast"]:
        plot[c] = plot[c].where(plot[c] > 0)
    long = plot.melt("Date", var_name="Series", value_name="Value").dropna()

    lines = (
        alt.Chart(long)
        .mark_line(strokeWidth=2.2, point=alt.OverlayMarkDef(size=22))
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Value:Q", title="Daily sales"),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(domain=["Actual", "Forecast"],
                                range=[ACTUAL_COLOR, FORECAST_COLOR]),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Series:N"),
                     alt.Tooltip("Value:Q", title="Sales", format=",.0f")],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"Date": [FORECAST_START]}))
        .mark_rule(strokeDash=[5, 5], color="#888888")
        .encode(x="Date:T")
    )
    return (lines + rule).properties(height=380).interactive()


def bar_chart(data, x, y, color, horizontal=False):
    if horizontal:
        enc = dict(
            y=alt.Y("%s:N" % x, sort="-x", title="Store"),
            x=alt.X("%s:Q" % y, title="Forecast total"),
        )
    else:
        enc = dict(
            x=alt.X("%s:N" % x, sort=data[x].tolist(), title=None),
            y=alt.Y("%s:Q" % y, title="Forecast total"),
        )
    return (
        alt.Chart(data)
        .mark_bar(color=color, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(tooltip=[alt.Tooltip("%s:N" % x), alt.Tooltip("%s:Q" % y, format=",.0f")],
                **enc)
        .properties(height=300)
    )


# ----------------------------------------------------------------------------- data
if not DATA_PATH.exists():
    st.error("app_data.csv was not found in the app folder. Run the last cell of "
             "notebook 06 to create it, then reload this page.")
    st.stop()

df = load_data(DATA_PATH.stat().st_mtime)

if df["Forecast"].notna().sum() == 0:
    st.error("app_data.csv has no forecast values. In notebook 06, rerun Cell 2 (forecast) "
             "and Cell 4 (build app data), then reload this page. Cell 4 needs "
             "data/processed/val_forecast.parquet from notebook 04.")
    st.stop()
if df.loc[df["Date"] >= FORECAST_START, "Forecast"].notna().sum() == 0:
    st.warning("No forecasts found for 1 Aug to 17 Sep 2015. Rerun notebook 06 Cell 2 and "
               "Cell 4 to create them.")

# ----------------------------------------------------------------------------- sidebar
st.sidebar.header("Controls")
store = st.sidebar.selectbox("Store", sorted(df["Store"].unique()),
                             help="Type a store number to jump to it.")
window = st.sidebar.radio(
    "Time window",
    ["Validation + forecast", "Forecast only", "Full history"],
    help="Validation = the 6 weeks the model was tested on (19 Jun to 31 Jul 2015).",
)
st.sidebar.caption("The dashed line in the charts marks where the forecast starts "
                   "(1 Aug 2015). Closed days are not plotted.")

forecast_dates = df.loc[df["Date"] >= FORECAST_START, "Date"]
if forecast_dates.empty:
    st.sidebar.warning("No forecast dates are available.")
    forecast_start = forecast_end = FORECAST_START
else:
    forecast_range = st.sidebar.slider(
        "Forecast date range",
        min_value=forecast_dates.min().date(),
        max_value=forecast_dates.max().date(),
        value=(forecast_dates.min().date(), forecast_dates.max().date()),
        format="MMM D, YYYY",
    )
    forecast_start, forecast_end = map(pd.Timestamp, forecast_range)

    what_if_promo = st.sidebar.toggle(
        "What-if promo (set Promo=1)",
        value=False,
        help="Re-predict forecast days assuming every store runs a promotion.",
    )

    if what_if_promo:
        missing_artifacts = [
            path.name for path in (MODEL_PATH, FEATURES_PATH) if not path.exists()
        ]
        if missing_artifacts:
            st.error(
                "What-if promo needs %s. Rerun notebook 06, including its final app-data "
                "cell, to create the model and forecast features."
                % " and ".join(missing_artifacts)
            )
            st.stop()

        model = load_model(MODEL_PATH.stat().st_mtime)
        feature_rows = load_forecast_features(FEATURES_PATH.stat().st_mtime)
        required_columns = {"Date", "Open", *model.feature_name_}
        missing_columns = required_columns.difference(feature_rows.columns)
        if missing_columns:
            st.error("Forecast feature file is missing columns: %s"
                     % ", ".join(sorted(missing_columns)))
            st.stop()
        if feature_rows.duplicated(["Store", "Date"]).any():
            st.error("Forecast feature file has duplicate Store/Date rows.")
            st.stop()

        prediction_features = feature_rows.copy()
        prediction_features["Promo"] = 1
        open_rows = prediction_features["Open"].eq(1)
        scenario_forecasts = prediction_features[["Store", "Date"]].copy()
        scenario_forecasts["Forecast"] = 0.0
        scenario_forecasts.loc[open_rows, "Forecast"] = np.expm1(
            model.predict(
                prediction_features.loc[open_rows, model.feature_name_]
            )
        )

        df = df.merge(
            scenario_forecasts,
            on=["Store", "Date"],
            how="left",
            suffixes=("", "_what_if"),
        )
        forecast_rows = df["Date"] >= FORECAST_START
        missing_forecasts = forecast_rows & df["Forecast_what_if"].isna()
        if missing_forecasts.any():
            st.error("What-if predictions are missing for some forecast dates.")
            st.stop()
        df.loc[forecast_rows, "Forecast"] = df.loc[forecast_rows, "Forecast_what_if"]
        df = df.drop(columns="Forecast_what_if")

    start = {"Validation + forecast": VALID_START,
         "Forecast only": FORECAST_START,
         "Full history": df["Date"].min()}[window]

store_df = df[df["Store"] == store].sort_values("Date").set_index("Date")
view = filter_forecast_range(store_df[store_df.index >= start],
                             forecast_start, forecast_end)
future = store_df[
    (store_df.index >= forecast_start) & (store_df.index <= forecast_end)
]

# ----------------------------------------------------------------------------- tabs
tab_store, tab_all, tab_insights, tab_model, tab_about = st.tabs(
    ["Store forecast", "All stores", "Insights", "Model insights", "About"])

# ---- Tab 1: one store
with tab_store:
    st.subheader("Store %s" % store)

    open_days = future["Forecast"][future["Forecast"] > 0]
    mape = compute_mape(store_df[(store_df.index >= VALID_START)
                                 & (store_df.index < FORECAST_START)])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecast total (selected dates)", fmt(future["Forecast"].sum()))
    c2.metric("Average per open day", fmt(open_days.mean()) if len(open_days) else "n/a")
    c3.metric("Busiest forecast day", fmt(open_days.max()) if len(open_days) else "n/a",
              help=(str(open_days.idxmax().date()) if len(open_days) else None))
    c4.metric("Test error (MAPE)", "n/a" if mape is None else "%.1f%%" % mape,
              help="Average percentage error on the 6 weeks the model had not seen.")

    st.markdown("##### Actual vs forecast")
    st.altair_chart(actual_vs_forecast_chart(view))

    left, right = st.columns([3, 2])
    with left:
        st.markdown("##### Forecast by week")
        wk = future[["Forecast"]].copy()
        wk["Week"] = wk.index.to_period("W").start_time
        wk = wk.groupby("Week", as_index=False)["Forecast"].sum()
        wk["Week starting"] = wk["Week"].dt.strftime("%d %b")
        st.altair_chart(bar_chart(wk, "Week starting", "Forecast", FORECAST_COLOR))
    with right:
        st.markdown("##### Download")
        csv = (future.reset_index()[["Date", "Forecast"]].dropna(subset=["Forecast"])
               .to_csv(index=False))
        st.download_button("Download this store's forecast (CSV)", csv.encode("utf-8"),
                           file_name="store_%s_forecast.csv" % store, mime="text/csv")

    st.markdown("##### Data table")
    which = st.radio("Show", ["Test period (actual vs forecast)",
                              "Future forecast (selected dates)"],
                     horizontal=True, key="table_view")
    if which.startswith("Test"):
        t = store_df[(store_df.index >= VALID_START) & (store_df.index < FORECAST_START)]
        t = t.reset_index().rename(columns={"Sales": "Actual"})
        t = t[(t["Actual"] > 0) & t["Forecast"].notna()].copy()   # open days only
        t["Difference"] = (t["Forecast"] - t["Actual"]).round(0)
        t["Error %"] = ((t["Forecast"] - t["Actual"]).abs() / t["Actual"] * 100).round(1)
        t["Actual"] = t["Actual"].round(0)
        t["Forecast"] = t["Forecast"].round(0)
        t["Day"] = t["Date"].dt.day_name()
        t["Date"] = t["Date"].dt.date
        if len(t) == 0:
            st.warning("app_data.csv has no forecasts for the test period "
                       "(19 Jun - 31 Jul 2015). Run `python rebuild_app_data.py` from the "
                       "project root (with the venv active), then reload this page.")
        else:
            st.dataframe(t[["Date", "Day", "Actual", "Forecast", "Difference", "Error %"]],
                         hide_index=True)
            st.caption("Average error on these days: %.1f%%. Closed days are left out. "
                       "Difference = forecast minus actual." % t["Error %"].mean())
    else:
        t = future.reset_index()[["Date", "Forecast"]].dropna().copy()
        t["Status"] = t["Forecast"].gt(0).map({True: "Open", False: "Closed"})
        t["Forecast"] = t["Forecast"].round(0)
        t["Day"] = t["Date"].dt.day_name()
        t["Date"] = t["Date"].dt.date
        st.dataframe(t[["Date", "Day", "Forecast", "Status"]], hide_index=True)
        st.caption("There are no actual sales for these dates: the dataset ends on "
                   "31 Jul 2015, so these are genuine forecasts that cannot be checked yet.")

# ---- Tab 2: all stores
with tab_all:
    st.subheader("All stores together")
    val_rows = df[(df["Date"] >= VALID_START) & (df["Date"] < FORECAST_START)]
    fut_rows = df[
        (df["Date"] >= forecast_start) & (df["Date"] <= forecast_end)
    ]
    overall_mape = compute_mape(val_rows)

    a1, a2, a3 = st.columns(3)
    a1.metric("Stores", fmt(df["Store"].nunique()))
    a2.metric("Forecast total (selected dates)", fmt(fut_rows["Forecast"].sum()))
    a3.metric("Test error (MAPE), all stores",
              "n/a" if overall_mape is None else "%.1f%%" % overall_mape)

    st.markdown("##### Total daily sales across all stores")
    totals = df.groupby("Date")[["Sales", "Forecast"]].sum(min_count=1)
    st.altair_chart(actual_vs_forecast_chart(
        filter_forecast_range(totals[totals.index >= start],
                              forecast_start, forecast_end)
    ))

    by_store = (fut_rows.groupby("Store")["Forecast"].sum()
                .sort_values(ascending=False))
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("##### Top 10 stores by forecast")
        top = by_store.head(10).reset_index()
        st.altair_chart(bar_chart(top, "Store", "Forecast", ACTUAL_COLOR, horizontal=True))
    with t2:
        st.markdown("##### Bottom 10 stores by forecast")
        bottom = by_store.tail(10).reset_index()
        st.altair_chart(bar_chart(bottom, "Store", "Forecast", FORECAST_COLOR,
                                  horizontal=True))

# ---- Tab 3: insights
with tab_insights:
    st.subheader("What drives sales, and what to do about it")

    rec = REPORT_DIR / "recommendations.md"
    if rec.exists():
        st.markdown(rec.read_text(encoding="utf-8"))
    else:
        st.info("Add reports/recommendations.md to show your business recommendations "
                "here.")

    charts = {
        "Total daily sales": "01_total_sales.png",
        "Weekday, month and promotion effects": "02_seasonality_promo.png",
        "Holidays, store types, competition": "03_holiday_storetype_competition.png",
        "Feature importance": "04_feature_importance.png",
        "Actual vs predicted (validation)": "06_actual_vs_pred.png",
    }
    available = {k: v for k, v in charts.items() if (FIG_DIR / v).exists()}
    if available:
        st.markdown("##### Analysis charts")
        choice = st.selectbox("Choose a chart", list(available))
        st.image(str(FIG_DIR / available[choice]))

# ---- Tab 4: model insights
with tab_model:
    st.subheader("Model explanation and comparison")

    shap_path = FIG_DIR / "05_shap_summary.png"
    if shap_path.exists():
        st.markdown("##### SHAP summary")
        st.image(str(shap_path))
    else:
        st.info("Add reports/figures/05_shap_summary.png to show the SHAP summary.")

    cmp_path = REPORT_DIR / "model_comparison.csv"
    if cmp_path.exists():
        st.markdown("##### Model comparison")
        st.dataframe(pd.read_csv(cmp_path), hide_index=True)
    else:
        st.info("Add reports/model_comparison.csv to show the model comparison.")

# ---- Tab 5: about
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
**Goal:** forecast daily sales for each Rossmann drugstore for the next 7 weeks, to help
with inventory planning and staff scheduling.

**Data:** Kaggle *Rossmann Store Sales* (about 1 million daily records for 1,115 stores,
Jan 2013 to Jul 2015) plus store details such as type, assortment, competition and
promotion schedules.

**Model:** LightGBM trained on log sales, using calendar features, promotions, store
attributes, store-level averages and lagged sales (all lags at least 49 days old, so
the model never uses information that would not exist at forecast time).

**Validation:** time-based. Trained up to 18 Jun 2015 and tested on the following 6 weeks,
then compared with simple baselines.

**How to use this dashboard**
1. Pick a store in the sidebar.
2. Use *Time window* and *Forecast date range* to adjust the dates shown.
3. Turn on *What-if promo* to re-predict with promotions enabled for forecast days.
4. Open *All stores* for the big picture, *Insights* for the findings, and
   *Model insights* for SHAP and model comparisons.

**Limitations:** forecasts for closed days are set to zero; promotions and holidays are
assumed known in advance; the model does not use weather or local events.
"""
    )