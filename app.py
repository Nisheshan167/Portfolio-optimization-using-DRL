import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

ASSETS = ["XLK", "XLF", "XLV", "XLE", "XLY", "EEM", "LQD", "IEF", "VNQ", "GLD", "SHY"]

# -----------------------------
# DATA
# -----------------------------
@st.cache_data
def download_prices(assets, start_date, end_date):
    data = yf.download(
        assets,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    # Case 1: MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        level0 = list(data.columns.get_level_values(0))
        level1 = list(data.columns.get_level_values(1))

        # Format like: ('Ticker', 'Close')
        if "Close" in level1:
            prices = data.xs("Close", axis=1, level=1)

        # Format like: ('Close', 'Ticker')
        elif "Close" in level0:
            prices = data.xs("Close", axis=1, level=0)

        else:
            raise ValueError(f"'Close' not found in MultiIndex columns: {data.columns}")

    # Case 2: Single-level columns
    else:
        if "Close" in data.columns:
            prices = data[["Close"]].copy()

            # If only one asset, rename Close -> ticker
            if len(assets) == 1:
                prices.columns = assets
        else:
            raise ValueError(f"'Close' not found in columns: {data.columns}")

    # Keep only requested assets that actually exist
    available_assets = [a for a in assets if a in prices.columns]
    missing_assets = [a for a in assets if a not in prices.columns]

    if missing_assets:
        st.warning(f"These assets were not returned and were skipped: {missing_assets}")

    prices = prices[available_assets]
    prices = prices.sort_index().ffill().dropna()

    if prices.empty:
        raise ValueError("Price dataframe is empty after cleaning.")

    return prices

@st.cache_data
def get_return_inputs(prices):
    returns = prices.pct_change().dropna()
    mean_daily_returns = returns.mean()
    cov_daily = returns.cov()

    annual_returns = mean_daily_returns * 252
    annual_cov = cov_daily * 252

    return returns, annual_returns, annual_cov

# -----------------------------
# PORTFOLIO FUNCTIONS
# -----------------------------
def portfolio_return(weights, annual_returns):
    return np.dot(weights, annual_returns)

def portfolio_volatility(weights, annual_cov):
    return np.sqrt(np.dot(weights.T, np.dot(annual_cov, weights)))

def sharpe_ratio(weights, annual_returns, annual_cov, risk_free_rate=0.0):
    port_ret = portfolio_return(weights, annual_returns)
    port_vol = portfolio_volatility(weights, annual_cov)
    if port_vol == 0:
        return 0
    return (port_ret - risk_free_rate) / port_vol

def max_return_for_target_risk(annual_returns, annual_cov, target_risk):
    n = len(annual_returns)
    init_guess = np.ones(n) / n
    bounds = tuple((0, 1) for _ in range(n))

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: portfolio_volatility(w, annual_cov) - target_risk}
    ]

    result = minimize(
        lambda w: -portfolio_return(w, annual_returns),
        init_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    return result

def min_risk_for_target_return(annual_returns, annual_cov, target_return):
    n = len(annual_returns)
    init_guess = np.ones(n) / n
    bounds = tuple((0, 1) for _ in range(n))

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: portfolio_return(w, annual_returns) - target_return}
    ]

    result = minimize(
        lambda w: portfolio_volatility(w, annual_cov),
        init_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    return result

def equal_weight_portfolio(annual_returns, annual_cov):
    n = len(annual_returns)
    weights = np.ones(n) / n
    ret = portfolio_return(weights, annual_returns)
    vol = portfolio_volatility(weights, annual_cov)
    sharpe = ret / vol if vol != 0 else np.nan
    return weights, ret, vol, sharpe

def compute_portfolio_timeseries(weights, returns_df):
    port_returns = returns_df.dot(weights)
    port_value = (1 + port_returns).cumprod()
    return port_returns, port_value

def compute_metrics(portfolio_returns, portfolio_values):
    cumulative_return = portfolio_values.iloc[-1] - 1
    annual_return = portfolio_returns.mean() * 252
    annual_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_volatility if annual_volatility != 0 else np.nan

    running_max = portfolio_values.cummax()
    drawdown = portfolio_values / running_max - 1
    max_drawdown = drawdown.min()

    return {
        "Cumulative Return": cumulative_return,
        "Annual Return": annual_return,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown
    }

# -----------------------------
# UI
# -----------------------------
st.title("Portfolio Optimizer")
st.write("Find the optimal portfolio for a chosen risk or return target, and compare it against an equal-weight portfolio.")

with st.sidebar:
    st.header("Inputs")

    start_date = st.date_input("Start Date", pd.to_datetime("2018-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2026-01-01"))

    objective = st.radio(
        "Optimization Goal",
        ["Highest return for chosen risk", "Lowest risk for chosen return"]
    )

    if objective == "Highest return for chosen risk":
        target_risk = st.slider("Target Annual Volatility", 0.03, 0.50, 0.12, 0.005)
    else:
        target_return = st.slider("Target Annual Return", 0.02, 0.23, 0.10, 0.005)

    run_btn = st.button("Optimize Portfolio")

if run_btn:
    try:
        with st.spinner("Downloading data..."):
            prices = download_prices(ASSETS, str(start_date), str(end_date))

        returns_df, annual_returns, annual_cov = get_return_inputs(prices)

        if objective == "Highest return for chosen risk":
            result = max_return_for_target_risk(annual_returns.values, annual_cov.values, target_risk)
        else:
            result = min_risk_for_target_return(annual_returns.values, annual_cov.values, target_return)

        if not result.success:
            st.error("Optimization failed. Try a different target risk/return.")
            st.stop()

        opt_weights = result.x
        opt_ret = portfolio_return(opt_weights, annual_returns.values)
        opt_vol = portfolio_volatility(opt_weights, annual_cov.values)
        opt_sharpe = opt_ret / opt_vol if opt_vol != 0 else np.nan

        eq_weights, eq_ret, eq_vol, eq_sharpe = equal_weight_portfolio(annual_returns.values, annual_cov.values)

        opt_port_returns, opt_port_values = compute_portfolio_timeseries(opt_weights, returns_df)
        eq_port_returns, eq_port_values = compute_portfolio_timeseries(eq_weights, returns_df)

        opt_metrics = compute_metrics(opt_port_returns, opt_port_values)
        eq_metrics = compute_metrics(eq_port_returns, eq_port_values)

        metrics_df = pd.DataFrame(
            [opt_metrics, eq_metrics],
            index=["Optimized Portfolio", "Equal Weight"]
        ).round(4)

        weights_df = pd.DataFrame({
            "Asset": ASSETS,
            "Optimized Weight": opt_weights
        }).sort_values("Optimized Weight", ascending=False)

        st.subheader("Optimal Weights")
        st.dataframe(weights_df.round(4), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Optimized Annual Return", f"{opt_ret:.2%}")
        c2.metric("Optimized Annual Volatility", f"{opt_vol:.2%}")
        c3.metric("Optimized Sharpe Ratio", f"{opt_sharpe:.2f}")

        st.subheader("Comparison with Equal-Weight Portfolio")
        st.dataframe(metrics_df, use_container_width=True)

        st.subheader("Growth of $1")
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(opt_port_values.index, opt_port_values.values, label="Optimized Portfolio")
        ax1.plot(eq_port_values.index, eq_port_values.values, label="Equal Weight Portfolio")
        ax1.set_title("Optimized vs Equal Weight Portfolio")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Portfolio Value")
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)

        st.subheader("Weight Allocation")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.bar(weights_df["Asset"], weights_df["Optimized Weight"])
        ax2.set_title("Optimized Portfolio Weights")
        ax2.set_xlabel("Asset")
        ax2.set_ylabel("Weight")
        ax2.grid(True, axis="y")
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Choose the user objective in the sidebar and click 'Optimize Portfolio'.")
