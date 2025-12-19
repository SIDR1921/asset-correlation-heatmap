import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from maincorr_clean import (
    DEFAULT_ASSETS, REGIME_PROXY, fetch_prices,
    compute_returns, rolling_correlation_matrices,
    detect_regimes, plot_corr_heatmap, build_snapshot, 
    mean_cross_asset_correlation, test_ticker_validity,
    explain_correlation_findings
)

st.sidebar.header("Correlation Analysis Settings")
start = st.sidebar.date_input("Start Date", value=dt.date(2019, 1, 1))
window = st.sidebar.slider("Rolling Window (Days)", 20, 250, 60, step=5)
returns_method = st.sidebar.selectbox("Return type", ["log", "simple"], index=0)

asset_groups = {}
for group, tickers in DEFAULT_ASSETS.items():
    st.sidebar.subheader(group)
    selected = st.sidebar.multiselect(
        f"Select {group} tickers", tickers, default=tickers
    )
    if selected:
        asset_groups[group] = selected

# Add correlation guide in sidebar
with st.sidebar:
    st.divider()
    st.subheader("📖 Quick Reference")
    st.markdown("""
    **Correlation Scale:**
    - 1.0 = Perfect positive (move together)
    - 0.5 = Moderate positive
    - 0.0 = No relationship
    - -0.5 = Moderate negative
    - -1.0 = Perfect negative (opposite moves)
    
    **Risk Levels:**
    - 🟢 0-30%: Great diversification
    - 🟡 30-70%: Moderate connection  
    - 🔴 70%+: High risk (move together)
    """)

if not asset_groups:
    st.warning("Please select at least one ticker from the sidebar.")
    st.stop()

st.header("Loading Data...")
with st.spinner("Fetching prices and computing correlations..."):
    try:
        tickers = [t for ts in asset_groups.values() for t in ts]
        st.write(f"🔄 Testing ticker validity...")
        
        # Test tickers first
        valid_tickers, invalid_tickers = test_ticker_validity(tickers, str(start))
        
        if invalid_tickers:
            st.warning(f"⚠️ Invalid/problematic tickers found: {', '.join(invalid_tickers)}")
        
        if not valid_tickers:
            st.error("❌ No valid tickers found. Please select different tickers or try an earlier start date.")
            st.stop()
        
        st.write(f"✅ Valid tickers: {', '.join(valid_tickers)}")
        
        # Update asset_groups to only include valid tickers
        updated_asset_groups = {}
        for group, group_tickers in asset_groups.items():
            valid_group_tickers = [t for t in group_tickers if t in valid_tickers]
            if valid_group_tickers:
                updated_asset_groups[group] = valid_group_tickers
        asset_groups = updated_asset_groups
        
        # Use only valid tickers
        prices = fetch_prices(valid_tickers, start=str(start))
        st.write(f"✅ Successfully fetched {len(prices)} data points")
        
        returns = compute_returns(prices, method=returns_method)
        st.write(f"✅ Computed returns: {len(returns)} data points")
        
        # Check if we have enough data for the selected window
        data_points = len(returns)
        if data_points < window:
            st.warning(f"Only {data_points} data points available, but window size is {window}.")
            new_window = max(20, min(data_points - 10, data_points // 2))
            if new_window < 20:
                st.error("Not enough data even for minimum window size of 20 days.")
                st.error("Please try selecting an earlier start date.")
                st.stop()
            st.info(f"Automatically reducing window size to {new_window}")
            window = new_window
        
        corr_mats = rolling_correlation_matrices(returns, window=window)
        vix = fetch_prices([REGIME_PROXY], start=str(start)).squeeze()
        equity_proxy = prices.get("^GSPC", prices.iloc[:, 0])
        regime = detect_regimes(vix, equity_proxy)
        dates = list(corr_mats.keys())
        
        # Update tickers list to only include valid ones for display
        tickers = valid_tickers
        
    except Exception as e:
        st.error(f"❌ Error fetching data: {str(e)}")
        st.error("**Troubleshooting suggestions:**")
        st.error("• Check your internet connection")
        st.error("• Try selecting an earlier start date (e.g., 2015-01-01)")
        st.error("• Try selecting fewer tickers or different tickers")
        st.error("• Some tickers might be delisted or invalid")
        
        # Show which tickers were being attempted
        if 'tickers' in locals():
            st.error(f"• Attempted tickers: {', '.join(tickers)}")
        st.stop()

if not dates:
    st.error("No correlation matrices could be computed.")
    st.error("This usually means there's insufficient data for the selected parameters.")
    st.stop()

# Data summary
st.success(f"✅ Successfully loaded {len(prices)} data points for {len(tickers)} tickers")
st.info(f"📊 Generated {len(corr_mats)} correlation matrices using {window}-day window")

with st.expander("📋 Data Summary"):
    st.write(f"**Date range:** {prices.index[0].date()} to {prices.index[-1].date()}")
    st.write(f"**Total trading days:** {len(prices)}")
    st.write(f"**Selected tickers:** {', '.join(tickers)}")
    st.write(f"**Window size:** {window} days")
    st.write(f"**Available correlation matrices:** {len(corr_mats)}")

sel_date = st.slider(
    "Select window end date", 
    min_value=dates[0].to_pydatetime().date(), 
    max_value=dates[-1].to_pydatetime().date(), 
    value=dates[-1].to_pydatetime().date(), 
    format="YYYY-MM-DD"
)
nearest_date = max(d for d in dates if d.date() <= sel_date)
corr_mat = corr_mats[nearest_date]

st.title("Cross-Asset Correlation Analysis")

st.header(f"Correlation Matrix - {nearest_date.date()}")
fig = plot_corr_heatmap(
    corr_mat, 
    title=f"{nearest_date.date()} | {window}-day Correlations"
)
st.pyplot(fig)

avg_cross = mean_cross_asset_correlation(corr_mat, asset_groups)
st.metric("Average Cross-Asset Correlation", f"{avg_cross:.3f}")

current_regime = regime.loc[nearest_date] if nearest_date in regime.index else "Unknown"
st.metric("Market Regime", current_regime)

# Generate and display simple explanations
st.header("📚 What This Means (In Simple Terms)")
explanations = explain_correlation_findings(corr_mat, asset_groups, avg_cross, current_regime)

for explanation in explanations:
    st.markdown(explanation)

st.divider()

with st.expander("View Raw Correlation Matrix"):
    st.dataframe(corr_mat.round(3))

with st.expander("View Price Data"):
    st.dataframe(prices.tail(10))


