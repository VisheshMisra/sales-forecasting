import numpy as np
import pandas as pd

LAGS = [49, 56, 63, 70, 364]
FEATURES = [
    "Store", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday", "StoreType",
    "Assortment", "CompetitionDistance", "CompMonthsOpen", "Promo2",
    "Promo2WeeksOpen", "Promo2Active", "Month", "Day", "WeekOfYear",
    "IsMonthStart", "IsMonthEnd", "lag_49", "lag_56", "lag_63", "lag_70", "lag_364",
    "lag_mean_49_70", "store_mean", "store_dow_mean", "store_promo_mean",
]


def clean_store_cols(df):
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
        df["CompetitionDistance"].max()
    )
    for column in [
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2SinceWeek",
        "Promo2SinceYear",
    ]:
        df[column] = df[column].fillna(0)
    df["PromoInterval"] = df["PromoInterval"].fillna("none")
    return df


def add_calendar(df):
    date = df["Date"]
    df["DayOfWeek"] = date.dt.dayofweek
    df["Month"] = date.dt.month
    df["Day"] = date.dt.day
    df["WeekOfYear"] = date.dt.isocalendar().week.astype(int)
    df["IsMonthStart"] = date.dt.is_month_start.astype(int)
    df["IsMonthEnd"] = date.dt.is_month_end.astype(int)
    return df


def add_store_features(df):
    # Months since competition opened, or 0 if no competition.
    year = df["CompetitionOpenSinceYear"].where(
        df["CompetitionOpenSinceYear"] > 0, 1900
    )
    month = df["CompetitionOpenSinceMonth"].where(
        df["CompetitionOpenSinceMonth"] > 0, 1
    )
    start = pd.to_datetime(
        dict(year=year.astype(int), month=month.astype(int), day=1)
    )
    months = (df["Date"] - start).dt.days / 30
    df["CompMonthsOpen"] = np.where(
        df["CompetitionOpenSinceYear"] > 0, months.clip(lower=0), 0
    )

    # Weeks since Promo2 started, and whether this month is a Promo2 month.
    year = df["Promo2SinceYear"].where(df["Promo2SinceYear"] > 0, 1900).astype(int)
    week = df["Promo2SinceWeek"].where(df["Promo2SinceWeek"] > 0, 1).astype(int)
    promo2_start = pd.to_datetime(year.astype(str) + "-01-01") + pd.to_timedelta(
        (week - 1) * 7, unit="D"
    )
    df["Promo2WeeksOpen"] = np.where(
        df["Promo2"] == 1,
        ((df["Date"] - promo2_start).dt.days / 7).clip(lower=0),
        0,
    )
    month_names = df["Date"].dt.strftime("%b").replace("Sep", "Sept")
    active = [
        int(month in interval) if interval != "none" else 0
        for month, interval in zip(month_names, df["PromoInterval"])
    ]
    df["Promo2Active"] = np.array(active) * df["Promo2"].values

    df["StoreType"] = df["StoreType"].map({"a": 0, "b": 1, "c": 2, "d": 3})
    df["Assortment"] = df["Assortment"].map({"a": 0, "b": 1, "c": 2})
    df["StateHoliday"] = df["StateHoliday"].astype(str).map(
        {"0": 0, "a": 1, "b": 2, "c": 3}
    )
    return df


def add_lags(df, src):
    # Shift dates, not rows, so lags remain correct when a store has gaps.
    for lag_days in LAGS:
        lag = src[["Store", "Date", "LogSales"]].copy()
        lag["Date"] = lag["Date"] + pd.Timedelta(days=lag_days)
        df = df.merge(
            lag.rename(columns={"LogSales": "lag_%d" % lag_days}),
            on=["Store", "Date"],
            how="left",
        )
    df["lag_mean_49_70"] = df[
        ["lag_49", "lag_56", "lag_63", "lag_70"]
    ].mean(axis=1)
    return df


def add_store_stats(df, src, cutoff):
    # Store-level averages use only days up to the cutoff to prevent leakage.
    history = src[src["Date"] <= cutoff]
    store_mean = (
        history.groupby("Store")["LogSales"].mean().rename("store_mean").reset_index()
    )
    store_dow_mean = (
        history.groupby(["Store", "DayOfWeek"])["LogSales"]
        .mean()
        .rename("store_dow_mean")
        .reset_index()
    )
    store_promo_mean = (
        history.groupby(["Store", "Promo"])["LogSales"]
        .mean()
        .rename("store_promo_mean")
        .reset_index()
    )
    df = df.merge(store_mean, on="Store", how="left")
    df = df.merge(store_dow_mean, on=["Store", "DayOfWeek"], how="left")
    df = df.merge(store_promo_mean, on=["Store", "Promo"], how="left")
    return df


def build_all(train, test, store, cutoff):
    # Combine training rows with test rows, which have no observed sales.
    test = test.copy()
    test["Sales"] = np.nan
    full = pd.concat([train, test.drop(columns="Id")], ignore_index=True)
    full["Open"] = full["Open"].fillna(1)
    full["StateHoliday"] = full["StateHoliday"].astype(str)
    full = full.merge(store, on="Store", how="left")
    full = clean_store_cols(full)
    full = add_calendar(full)
    full = add_store_features(full)
    full["LogSales"] = np.log1p(full["Sales"])
    source = full[(full["Open"] == 1) & (full["Sales"] > 0)]
    full = add_lags(full, source)
    source = full[(full["Open"] == 1) & (full["Sales"] > 0)]
    full = add_store_stats(full, source, pd.Timestamp(cutoff))
    return full


