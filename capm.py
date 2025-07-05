import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
from pypfopt import EfficientFrontier, risk_models, expected_returns
import streamlit as st

# Define assets
asset_tickers = [
    "AIA.AX", "ALL.AX", "AMC.AX", "ANZ.AX", "APA.AX", "ASX.AX",
    "BHP.AX", "BSL.AX", "BXB.AX", "CBA.AX", "COH.AX", "COL.AX", "CSL.AX",
    "DXS.AX", "FMG.AX", "FPH.AX", "GMG.AX", "IAG.AX", "JHX.AX", "MGOC.AX",
    "MGR.AX", "MQG.AX", "NAB.AX", "NST.AX", "QBE.AX", "REA.AX",
    "REH.AX", "RHC.AX", "RIO.AX", "RMD.AX", "S32.AX", "SCG.AX", "SEK.AX",
    "SGP.AX", "SHL.AX", "STO.AX", "SUN.AX", "TAH.AX", "TCL.AX",
    "TLS.AX", "TPG.AX", "WBC.AX", "WES.AX", "WOW.AX", "WTC.AX", "XRO.AX"
]
market_ticker = "^AXJO"


@st.cache_data
def finance_download_cache(ticker_list, start, end):
    data = yf.download(ticker_list, start=start, end=end, auto_adjust=False)["Adj Close"]
    if data is None or data.empty:
        raise ValueError("No data downloaded from Yahoo")
    return data


def finance_download_csv(ticker_list, start, end, file_name):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    if os.path.exists(file_name):
        print(f"Found data in: {file_name}.")
        data = pd.read_csv(file_name, index_col=0, parse_dates=True)
    else:
        data = pd.DataFrame()

    existing_tickers = data.columns.tolist() if not data.empty else []
    missing_tickers = list(set(ticker_list) - set(existing_tickers))

    if data.empty:
        print(f"Downloading full data for {ticker_list} ...")
        new_data = yf.download(ticker_list, start=start, end=end, auto_adjust=False)["Adj Close"]
        if new_data is None or new_data.empty:
            raise ValueError("No data downloaded - check tickers or network connection")
        new_data.to_csv(file_name)
        return new_data

    earliest = data.index.min()
    latest = data.index.max()

    if start < earliest:
        print(f"Downloading earlier data from {start.date()} to {(earliest - pd.Timedelta(days=1)).date()}")
        prev_data = yf.download(
            existing_tickers,
            start=start.strftime("%Y-%m-%d"),
            end=(earliest - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=False
        )["Adj Close"]
        if prev_data is not None and not prev_data.empty:
            data = pd.concat([data, prev_data])
            data = data[~data.index.duplicated(keep='first')]

    if end > latest:
        print(f"Downloading updates from {(latest + pd.Timedelta(days=1)).date()} to {end.date()}")
        next_data = yf.download(
            existing_tickers,
            start=(latest + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=False
        )["Adj Close"]
        if next_data is not None and not next_data.empty:
            data = pd.concat([data, next_data])
            data = data[~data.index.duplicated(keep='first')]

    if missing_tickers:
        print(f"Missing tickers: {missing_tickers}...Downloading")
        missing_data = \
        yf.download(missing_tickers, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), auto_adjust=False)[
            "Adj Close"]
        if missing_data is not None and not missing_data.empty:
            if not data.empty:
                missing_data = missing_data.reindex(data.index)
                missing_data = missing_data.ffill()
            data = pd.concat([data, missing_data], axis=1)
        else:
            print("Warning: no data found for missing tickers.")

    data = data.loc[:, ~data.columns.duplicated(keep='first')]
    data = data.ffill().sort_index()
    data.to_csv(file_name)
    print(f"Data saved to CSV {file_name} with shape {data.shape}.")
    return data


def finance_download(ticker_list, start, end, file_name, use_cache=True):
    if use_cache:
        return finance_download_cache(ticker_list, start, end)
    else:
        return finance_download_csv(ticker_list, start, end, file_name)


# CAPM
def capm_analysis(asset_data: pd.DataFrame, market_ticker: str, risk_free_rate=0.03, expected_market_return=0.08):
    returns = asset_data.pct_change(fill_method=None).dropna()
    if market_ticker not in returns.columns:
        raise ValueError(f"Market ticker {market_ticker} is not in returns")

    market_returns = returns[market_ticker]
    results = []

    for ticker in returns.columns:
        if ticker == market_ticker:
            continue
        y = returns[ticker]
        x = sm.add_constant(market_returns)
        model = sm.OLS(y, x).fit()
        alpha = model.params.iloc[0]
        beta = model.params.iloc[1]
        expected_return = risk_free_rate + beta * (expected_market_return - risk_free_rate)
        results.append({
            "Ticker": ticker,
            "Alpha": alpha,
            "Beta": beta,
            "Expected Return (CAPM)": expected_return
        })
    capm_summary = pd.DataFrame(results)
    capm_summary.to_csv("data/capm_results.csv", index=False)
    st.success("CAPM results saved to data/capm_results.csv")
    return capm_summary


def plot_sml(capm_df, risk_free_rate=0.03, expected_market_return=0.08, file_name="sml_plot.png"):
    betas = np.linspace(0, 2, 20)
    sml_returns = risk_free_rate + betas * (expected_market_return - risk_free_rate)

    plt.figure(figsize=(10, 6))
    plt.plot(betas, sml_returns, label="Security Market Line", color="blue")

    top_assets = capm_df.sort_values("Expected Return (CAPM)", ascending=False).head(10)
    for _, row in top_assets.iterrows():
        plt.scatter(row["Beta"], row["Expected Return (CAPM)"], marker="o", s=80)
        plt.text(row["Beta"] + 0.03, row["Expected Return (CAPM)"] + 0.001, row["Ticker"], fontsize=6, rotation=30)

    plt.xlabel("Beta")
    plt.ylabel("Expected Return (CAPM)")
    plt.title("Security Market Line (Top 10 ASX50 Assets)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"SML plot saved as {file_name}")


def optimised_weights(asset_data, max_assets=5, min_weight=0.05, risk_free_rate=0.03):
    mu = expected_returns.mean_historical_return(asset_data)
    S = risk_models.sample_cov(asset_data)

    ef = EfficientFrontier(mu, S)
    ef.add_constraint(lambda w: w >= 0)
    ef.max_sharpe(risk_free_rate=risk_free_rate)
    cleaned_weights = ef.clean_weights()

    sorted_weights = dict(sorted(cleaned_weights.items(), key=lambda x: x[1], reverse=True))
    top_weights = dict(list(sorted_weights.items())[:max_assets])
    total = sum(top_weights.values())
    top_weights = {k: v / total for k, v in top_weights.items()}

    for k, v in top_weights.items():
        if v < min_weight:
            print(f"Warning: {k} has weight {v:.2%}, below minimum 5%")

    performance = ef.portfolio_performance(verbose=True)

    weights_df = pd.DataFrame.from_dict(top_weights, orient="index", columns=["Weight"])
    weights_df.to_csv("data/optimised_weights.csv")
    st.success("Optimised weights saved to data/optimised_weights.csv")

    print("Top assets:", top_weights)
    return top_weights, performance


def plot_portfolio_pie(weights: dict):
    filtered_weights = {k: v for k, v in weights.items() if v > 0.02}
    labels = list(filtered_weights.keys())
    sizes = list(filtered_weights.values())

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.85
    )
    ax.set_title("Optimised Portfolio Allocation (Top Holdings)")
    st.pyplot(fig)
    print("Portfolio pie chart displayed directly in Streamlit")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_assets", type=int, default=5, help="Number of top assets to include in the portfolio")
    args = parser.parse_args()

    start = "2020-01-01"
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    print(f"Updating data from {start} to {end}")

    data = finance_download(
        ticker_list=asset_tickers + [market_ticker],
        start=start,
        end=end,
        file_name="data/asx50_data.csv",
        use_cache=False
    )
    print("Local CSV updated successfully.")

    # CAPM
    capm_df = capm_analysis(data, market_ticker)
    print("CAPM results updated locally.")

    # Portfolio optimiser
    weights, perf = optimised_weights(data, max_assets=args.max_assets)
    print("Optimised weights updated locally.")