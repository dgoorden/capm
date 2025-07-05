import streamlit as st
import pandas as pd
from capm import (
    finance_download,
    capm_analysis,
    plot_sml,
    optimised_weights,
    plot_portfolio_pie,
    asset_tickers,
    market_ticker,
)

st.title("ASX50 CAPM & Portfolio Optimiser")

# Sidebar controls
start_date = st.sidebar.date_input("Start date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End date", pd.to_datetime("2022-12-31"))
risk_free_rate = st.sidebar.slider("Risk-free rate", 0.0, 0.1, 0.03)
expected_market_return = st.sidebar.slider("Expected market return", 0.0, 0.15, 0.08)
min_weight = st.sidebar.slider("Minimum asset weight", 0.0, 0.2, 0.05)

# Main actions
if st.button("Run Analysis"):
    st.write("Downloading and preparing data...")
    data = finance_download(
        ticker_list=asset_tickers + [market_ticker],
        start=start_date,
        end=end_date,
        file_name=".venv/asx50_data.csv"
    )
    st.success("Data loaded successfully!")

    st.write("Running CAPM analysis...")
    capm_df = capm_analysis(data, market_ticker, risk_free_rate, expected_market_return)
    st.dataframe(capm_df)

    plot_sml(capm_df)
    st.image("sml_plot.png")

    st.write("Optimising portfolio...")
    weights, perf = optimised_weights(data, min_weight=min_weight)
    weights_df = pd.DataFrame.from_dict(weights, orient="index", columns=["Weight"])
    st.dataframe(weights_df)

    st.image("portfolio_pie.png")
    st.write(f"Expected return: {perf[0]:.2%}")
    st.write(f"Volatility: {perf[1]:.2%}")
    st.write(f"Sharpe ratio: {perf[2]:.2f}")

    st.success("Analysis complete.")