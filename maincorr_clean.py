import datetime as dt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
from typing import List, Dict, Optional, Tuple

DEFAULT_ASSETS = {
    "Equities": ["^GSPC", "^NDX", "^STOXX50E"],
    "FX": ["EURUSD=X", "USDJPY=X", "GBPUSD=X"],
    "Rates": ["^TNX", "^IRX"],
    "Commodities": ["GC=F", "CL=F", "HG=F"],
}
REGIME_PROXY = "^VIX"

def fetch_prices(
    tickers: List[str],
    start: str = "2019-01-01",
    end: Optional[str] = None
) -> pd.DataFrame:
    end = end or dt.date.today().strftime("%Y-%m-%d")
    
    # Try downloading with different parameters for better reliability
    try:
        data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True, prepost=False)
    except Exception as e:
        raise ValueError(f"Failed to download data for {tickers}: {str(e)}")
    
    if data is None or data.empty:
        raise ValueError(f"No data found for tickers: {tickers}. Check if tickers are valid and date range is appropriate.")
    
    # Handle different data structures from yfinance
    if len(tickers) == 1:
        # Single ticker - data might be a Series or DataFrame
        if "Adj Close" in data.columns:
            data = data[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
        elif "Close" in data.columns:
            data = data[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            # Take the last column if structure is unexpected
            data = data.iloc[:, [-1]]
            data.columns = tickers
    else:
        # Multiple tickers - MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            if "Adj Close" in data.columns.levels[0]:
                data = data["Adj Close"]
            elif "Close" in data.columns.levels[0]:
                data = data["Close"]
            else:
                # Take a reasonable column
                data = data.iloc[:, :len(tickers)]
        else:
            # Fallback for unexpected structure
            data = data.iloc[:, :len(tickers)]
    
    # Ensure we have a DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()
    
    # Clean the data
    data = data.dropna(how='all')  # Remove rows where all values are NaN
    data = data.ffill().bfill()  # Forward fill then backward fill
    
    if data.empty:
        raise ValueError(f"All data is NaN for tickers: {tickers}")
    
    return data.sort_index()

def test_ticker_validity(tickers: List[str], start: str) -> Tuple[List[str], List[str]]:
    """Test individual tickers and return valid and invalid ones."""
    valid_tickers = []
    invalid_tickers = []
    
    for ticker in tickers:
        try:
            test_data = yf.download(ticker, start=start, period="1mo", progress=False)
            if not test_data.empty and len(test_data) > 5:
                valid_tickers.append(ticker)
            else:
                invalid_tickers.append(ticker)
        except:
            invalid_tickers.append(ticker)
    
    return valid_tickers, invalid_tickers

def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    if method == "log":
        rets = np.log(prices / prices.shift(1))
    elif method == "simple":
        rets = prices.pct_change()
    else:
        raise ValueError("method must be 'log' or 'simple'")
    return rets.dropna(how="all")

def rolling_correlation_matrices(
    returns: pd.DataFrame, 
    window: int = 60
) -> Dict[pd.Timestamp, pd.DataFrame]:
    if len(returns) < window:
        raise ValueError(f"Not enough data points ({len(returns)}) for window size {window}. Need at least {window} data points.")
    
    mats = {}
    for end_idx in range(window - 1, len(returns)):
        window_slice = returns.iloc[end_idx - window + 1:end_idx + 1]
        corr_mat = window_slice.corr()
        if not corr_mat.empty:
            mats[returns.index[end_idx]] = corr_mat
    return mats

def detect_regimes(
    vix: pd.Series, 
    equity_proxy: pd.Series, 
    lookback: int = 5, 
    vix_high: float = 25.0, 
    eq_return_threshold: float = -0.02
) -> pd.Series:
    eq_short = equity_proxy.pct_change(lookback)
    regime = pd.Series(index=vix.index, dtype="object")
    regime[(vix >= vix_high) & (eq_short <= eq_return_threshold)] = "Risk-Off"
    regime[(vix < vix_high) & (eq_short > 0.0)] = "Risk-On"
    regime[regime.isna()] = "Neutral"
    return regime

def plot_corr_heatmap(
    corr_matrix: pd.DataFrame, 
    title: str = "Rolling Correlation Matrix", 
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr_matrix, 
        cmap="coolwarm", 
        vmin=-1.0, 
        vmax=1.0, 
        center=0.0, 
        annot=False, 
        linewidths=0.5, 
        cbar_kws={"label": "Correlation"}, 
        ax=ax
    )
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    return fig

def mean_cross_asset_correlation(
    corr_matrix: pd.DataFrame, 
    asset_groups: Dict[str, List[str]]
) -> float:
    cols = corr_matrix.columns
    ticker_to_group = {t: g for g, ts in asset_groups.items() for t in ts}
    cross_pairs = []
    
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j <= i:
                continue
            ga, gb = ticker_to_group.get(a), ticker_to_group.get(b)
            if ga and gb and ga != gb:
                cross_pairs.append(corr_matrix.loc[a, b])
    
    return float(np.nanmean(cross_pairs)) if cross_pairs else np.nan

def build_snapshot(
    asset_groups: Dict[str, List[str]] = DEFAULT_ASSETS,
    start: str = "2019-01-01",
    window: int = 60,
    returns_method: str = "log",
) -> Dict:
    tickers = [t for ts in asset_groups.values() for t in ts]
    prices = fetch_prices(tickers, start=start)
    rets = compute_returns(prices, method=returns_method)
    mats = rolling_correlation_matrices(rets, window=window)
    
    if not mats:
        raise RuntimeError("No correlation matrices computed.")
    
    vix = fetch_prices([REGIME_PROXY], start=start).squeeze()
    equity_proxy = prices.get("^GSPC", prices.iloc[:, 0])
    regime = detect_regimes(vix, equity_proxy)
    
    latest_date = list(mats.keys())[-1]
    latest_mat = mats[latest_date]
    fig = plot_corr_heatmap(
        latest_mat, 
        title=f"{latest_date.date()} | {window}-day Correlations"
    )
    avg_cross = mean_cross_asset_correlation(latest_mat, asset_groups)
    
    # Get current regime for the latest date
    current_regime = regime.loc[latest_date] if latest_date in regime.index else "Unknown"
    
    # Generate simple explanations
    explanations = explain_correlation_findings(latest_mat, asset_groups, avg_cross, current_regime)
    
    return {
        "prices": prices,
        "returns": rets,
        "corr_mats": mats,
        "latest_date": latest_date,
        "latest_matrix": latest_mat,
        "regime_series": regime,
        "avg_cross_corr": avg_cross,
        "figure": fig,
        "explanations": explanations,
    }

def explain_correlation_findings(
    corr_matrix: pd.DataFrame, 
    asset_groups: Dict[str, List[str]],
    avg_cross_corr: float,
    regime: str = "Unknown"
) -> List[str]:
    """
    Generate simple, easy-to-understand explanations of correlation findings.
    
    Args:
        corr_matrix: Correlation matrix
        asset_groups: Dictionary mapping asset group names to ticker lists
        avg_cross_corr: Average cross-asset correlation
        regime: Current market regime
    
    Returns:
        List of simple explanation strings
    """
    explanations = []
    
    # Overall correlation level
    if avg_cross_corr > 0.7:
        explanations.append("🔴 **HIGH RISK**: Assets are moving very similarly (70%+ correlation). When one falls, most others will likely fall too.")
    elif avg_cross_corr > 0.5:
        explanations.append("🟡 **MODERATE RISK**: Assets are somewhat connected (50-70% correlation). Diversification benefits are limited.")
    elif avg_cross_corr > 0.3:
        explanations.append("🟢 **GOOD DIVERSIFICATION**: Assets have moderate connections (30-50% correlation). This is healthy for portfolio spread.")
    elif avg_cross_corr > 0:
        explanations.append("🟢 **EXCELLENT DIVERSIFICATION**: Assets move quite independently (0-30% correlation). Great for risk reduction.")
    else:
        explanations.append("🔵 **NEGATIVE CORRELATION**: Some assets move in opposite directions. This can provide natural hedging.")
    
    # Market regime context
    if regime == "Risk-Off":
        explanations.append("⚠️ **RISK-OFF MARKET**: During stressful times like now, correlations often increase as investors panic-sell everything.")
    elif regime == "Risk-On":
        explanations.append("📈 **RISK-ON MARKET**: In good times like now, assets may move more independently as investors pick specific opportunities.")
    
    # Find strongest and weakest correlations
    if not corr_matrix.empty and len(corr_matrix) > 1:
        # Get upper triangle of correlation matrix (avoid duplicates)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        correlations = corr_matrix.where(mask).stack().dropna()
        
        if not correlations.empty:
            strongest_corr = correlations.max()
            weakest_corr = correlations.min()
            strongest_pair = correlations.idxmax()
            weakest_pair = correlations.idxmin()
            
            # Strongest correlation
            if strongest_corr > 0.8:
                explanations.append(f"🔗 **STRONGEST CONNECTION**: {strongest_pair[0]} and {strongest_pair[1]} are highly connected ({strongest_corr:.1%}). They usually move together.")
            
            # Weakest/most negative correlation
            if weakest_corr < -0.3:
                explanations.append(f"↔️ **NATURAL HEDGE**: {weakest_pair[0]} and {weakest_pair[1]} often move in opposite directions ({weakest_corr:.1%}). One can protect against the other.")
            elif weakest_corr < 0.2:
                explanations.append(f"🔀 **INDEPENDENT MOVERS**: {weakest_pair[0]} and {weakest_pair[1]} move quite independently ({weakest_corr:.1%}). Good for diversification.")
    
    # Asset group analysis
    group_correlations = {}
    for group1, tickers1 in asset_groups.items():
        for group2, tickers2 in asset_groups.items():
            if group1 < group2:  # Avoid duplicates
                group_tickers1 = [t for t in tickers1 if t in corr_matrix.columns]
                group_tickers2 = [t for t in tickers2 if t in corr_matrix.columns]
                
                if group_tickers1 and group_tickers2:
                    correlations_between_groups = []
                    for t1 in group_tickers1:
                        for t2 in group_tickers2:
                            correlations_between_groups.append(corr_matrix.loc[t1, t2])
                    
                    if correlations_between_groups:
                        avg_group_corr = np.mean(correlations_between_groups)
                        group_correlations[f"{group1}-{group2}"] = avg_group_corr
    
    # Explain group relationships
    if group_correlations:
        highest_group_corr = max(group_correlations.items(), key=lambda x: x[1])
        lowest_group_corr = min(group_correlations.items(), key=lambda x: x[1])
        
        if highest_group_corr[1] > 0.6:
            explanations.append(f"🏢 **CONNECTED SECTORS**: {highest_group_corr[0].replace('-', ' and ')} are highly connected ({highest_group_corr[1]:.1%}). They tend to rise and fall together.")
        
        if lowest_group_corr[1] < 0.2:
            explanations.append(f"🌐 **GOOD MIX**: {lowest_group_corr[0].replace('-', ' and ')} work well together ({lowest_group_corr[1]:.1%}). They provide good diversification.")
    
    # Practical advice
    if avg_cross_corr > 0.6:
        explanations.append("💡 **ADVICE**: Consider reducing position sizes during high correlation periods. All investments might fall together.")
        explanations.append("💡 **TIP**: Look for assets outside these groups or consider cash/bonds as safe havens.")
    elif avg_cross_corr < 0.3:
        explanations.append("💡 **ADVICE**: This is a great time for diversified investing. Your risks are well spread.")
        explanations.append("💡 **TIP**: You can potentially take slightly larger positions since risks are distributed.")
    
    return explanations

if __name__ == "__main__":
    snapshot = build_snapshot(window=60)
    print(f"Latest date: {snapshot['latest_date']}")
    print(f"Avg cross-asset correlation: {snapshot['avg_cross_corr']:.3f}")
    
    print("\n" + "="*50)
    print("CORRELATION ANALYSIS EXPLAINED:")
    print("="*50)
    
    for explanation in snapshot['explanations']:
        # Remove markdown formatting for console output
        clean_explanation = explanation.replace("**", "").replace("🔴", "").replace("🟡", "").replace("🟢", "").replace("🔵", "").replace("⚠️", "").replace("📈", "").replace("🔗", "").replace("↔️", "").replace("🔀", "").replace("🏢", "").replace("🌐", "").replace("💡", "").replace("📋", "")
        print(f"• {clean_explanation}")
    
    snapshot["figure"].show()
