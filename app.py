import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

st.set_page_config(page_title="DRL Portfolio Optimization", layout="wide")

# -----------------------------
# CONFIG
# -----------------------------
ASSETS = ["XLK", "XLF", "XLV", "XLE", "XLY", "EEM", "LQD", "IEF", "VNQ", "GLD", "SHY"]

# -----------------------------
# ENVIRONMENT
# -----------------------------
class PortfolioEnv(gym.Env):
    def __init__(self, features_df, returns_df, asset_names, transaction_cost=0.001):
        super(PortfolioEnv, self).__init__()

        self.features_df = features_df.copy()
        self.returns_df = returns_df.copy()
        self.asset_names = asset_names
        self.n_assets = len(asset_names)
        self.transaction_cost = transaction_cost

        self.returns_df = self.returns_df.loc[self.features_df.index]

        self.features = self.features_df.values
        self.asset_returns = self.returns_df[self.asset_names].values
        self.dates = self.features_df.index

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )

        self.n_features = self.features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features + self.n_assets,),
            dtype=np.float32
        )

        self.reset()

    def _normalize_weights(self, action):
        action = np.clip(action, 0, 1)
        total = np.sum(action)
        if total == 0:
            return np.ones(self.n_assets) / self.n_assets
        return action / total

    def _get_observation(self):
        current_features = self.features[self.current_step]
        obs = np.concatenate([current_features, self.prev_weights])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.prev_weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = 1.0
        self.portfolio_history = [self.portfolio_value]
        self.weight_history = [self.prev_weights.copy()]
        self.reward_history = []

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        weights = self._normalize_weights(action)
        current_asset_returns = self.asset_returns[self.current_step]

        portfolio_return = np.dot(weights, current_asset_returns)
        turnover = np.sum(np.abs(weights - self.prev_weights))
        cost = self.transaction_cost * turnover
        reward = portfolio_return - cost

        self.portfolio_value *= (1 + reward)

        self.portfolio_history.append(self.portfolio_value)
        self.weight_history.append(weights.copy())
        self.reward_history.append(reward)

        self.prev_weights = weights
        self.current_step += 1

        terminated = self.current_step >= len(self.features) - 1
        truncated = False

        if not terminated:
            observation = self._get_observation()
        else:
            observation = np.zeros(self.n_features + self.n_assets, dtype=np.float32)

        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return,
            "transaction_cost": cost,
            "turnover": turnover,
            "weights": weights
        }

        return observation, reward, terminated, truncated, info

# -----------------------------
# HELPERS
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

    prices = data.xs("Close", axis=1, level="Price")
    prices = prices[assets]
    prices = prices.sort_index()
    prices = prices.dropna(how="all")
    prices = prices.ffill().dropna()
    return prices

@st.cache_data
def build_features(prices, start_date, end_date):
    asset_returns = prices.pct_change().dropna()
    volatility = asset_returns.rolling(window=20).std()
    momentum = prices.pct_change(periods=20)

    vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)

    if isinstance(vix.columns, pd.MultiIndex):
        vix = vix.xs("Close", axis=1, level=-1).squeeze()
    else:
        vix = vix["Close"]

    vix = vix.reindex(prices.index)
    vix = vix.ffill()

    returns_feat = asset_returns.copy()
    volatility_feat = volatility.copy()
    momentum_feat = momentum.copy()

    returns_feat.columns = [f"{col}_ret" for col in returns_feat.columns]
    volatility_feat.columns = [f"{col}_vol" for col in volatility_feat.columns]
    momentum_feat.columns = [f"{col}_mom" for col in momentum_feat.columns]

    features = pd.concat([returns_feat, volatility_feat, momentum_feat], axis=1)
    features["VIX"] = vix
    features = features.dropna()

    asset_returns = asset_returns.loc[features.index]
    return features, asset_returns

def compute_metrics(portfolio_series, freq=252):
    returns = portfolio_series.pct_change().dropna()

    cumulative_return = portfolio_series.iloc[-1] / portfolio_series.iloc[0] - 1
    annual_return = (1 + cumulative_return) ** (freq / len(returns)) - 1
    annual_vol = returns.std() * np.sqrt(freq)
    sharpe = annual_return / annual_vol if annual_vol != 0 else np.nan

    running_max = portfolio_series.cummax()
    drawdown = portfolio_series / running_max - 1
    max_drawdown = drawdown.min()

    return {
        "Cumulative Return": cumulative_return,
        "Annual Return": annual_return,
        "Annual Volatility": annual_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown
    }

def run_model_on_env(model, env):
    obs, info = env.reset()
    done = False
    portfolio_values = [env.portfolio_value]
    weights_history = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        portfolio_values.append(info["portfolio_value"])
        weights_history.append(info["weights"])

    return portfolio_values, weights_history

# -----------------------------
# UI
# -----------------------------
st.title("DRL Portfolio Optimization Dashboard")
st.write("PPO-based portfolio allocation across sector ETFs, bonds, gold, REITs, emerging markets, and cash.")

with st.sidebar:
    st.header("Settings")
    start_date = st.date_input("Start Date", pd.to_datetime("2014-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2026-01-01"))
    split_ratio = st.slider("Train/Test Split", min_value=0.6, max_value=0.9, value=0.8, step=0.05)
    transaction_cost = st.number_input("Transaction Cost", min_value=0.0, max_value=0.01, value=0.001, step=0.0005)
    model_path = st.text_input("Saved PPO Model Path", value="ppo_portfolio_model.zip")
    run_btn = st.button("Run Analysis")

if run_btn:
    try:
        with st.spinner("Downloading price data..."):
            prices = download_prices(ASSETS, str(start_date), str(end_date))

        with st.spinner("Building features..."):
            features, asset_returns = build_features(prices, str(start_date), str(end_date))

        split_idx = int(len(features) * split_ratio)

        train_features = features.iloc[:split_idx]
        test_features = features.iloc[split_idx:]

        train_returns = asset_returns.iloc[:split_idx]
        test_returns = asset_returns.iloc[split_idx:]

        test_env = PortfolioEnv(
            features_df=test_features,
            returns_df=test_returns,
            asset_names=ASSETS,
            transaction_cost=transaction_cost
        )

        with st.spinner("Loading trained PPO model..."):
            model = PPO.load(model_path)

        with st.spinner("Running DRL portfolio..."):
            drl_values, weights_history = run_model_on_env(model, test_env)

        drl_series = pd.Series(drl_values, name="DRL").reset_index(drop=True)

        eq_weights = np.ones(len(ASSETS)) / len(ASSETS)
        eq_returns = (test_returns[ASSETS] * eq_weights).sum(axis=1)
        eq_values = (1 + eq_returns).cumprod()
        eq_series = pd.Series(eq_values.values, name="Equal Weight").reset_index(drop=True)

        min_len = min(len(drl_series), len(eq_series))
        drl_series = drl_series.iloc[:min_len]
        eq_series = eq_series.iloc[:min_len]

        drl_metrics = compute_metrics(drl_series)
        eq_metrics = compute_metrics(eq_series)

        metrics_df = pd.DataFrame([drl_metrics, eq_metrics], index=["DRL", "Equal Weight"])
        metrics_df = metrics_df.applymap(lambda x: round(x, 4) if pd.notnull(x) else x)

        st.subheader("Performance Metrics")
        st.dataframe(metrics_df, use_container_width=True)

        st.subheader("Portfolio Value Comparison")
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(drl_series, label="DRL Portfolio")
        ax1.plot(eq_series, label="Equal Weight Portfolio")
        ax1.set_title("DRL vs Equal Weight")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Portfolio Value")
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)

        weights_df = pd.DataFrame(weights_history, columns=ASSETS)

        st.subheader("Portfolio Weights Over Time")
        fig2, ax2 = plt.subplots(figsize=(14, 6))
        for col in weights_df.columns:
            ax2.plot(weights_df[col], label=col)
        ax2.set_title("DRL Portfolio Weights")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Weight")
        ax2.legend(loc="upper right", ncol=3)
        ax2.grid(True)
        st.pyplot(fig2)

        st.subheader("Latest Recommended Weights")
        latest_weights = pd.DataFrame({
            "Asset": ASSETS,
            "Weight": weights_df.iloc[-1].values
        }).sort_values("Weight", ascending=False)
        latest_weights["Weight"] = latest_weights["Weight"].round(4)
        st.dataframe(latest_weights, use_container_width=True)

    except FileNotFoundError:
        st.error("Saved PPO model file not found. Make sure the model path is correct.")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Set your parameters in the sidebar and click Run Analysis.")
