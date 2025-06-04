import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set Streamlit config
st.set_page_config(page_title="Nifty50 Data Explorer", layout="wide")

# Sidebar
st.sidebar.title("📊 Nifty50 Data Explorer")
section = st.sidebar.radio("Choose Section", [
    "Dataset Overview",
    "Data Shape",
    "Missing Values",
    "Descriptive Stats",
    "Portfolio Simulation"
])

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("nifty50_closing_prices.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    return df

df_nifty = load_data()

# Prepare stocks list (exclude Date column)
stocks = [col for col in df_nifty.columns if col != 'Date']

# Prepare descriptive stats DataFrame excluding 'HDFC.NS' due to missing values
df_stats = df_nifty[stocks].drop('HDFC.NS', axis=1)

# Descriptive stats
mean_prices = df_stats.mean()
std_prices = df_stats.std()
max_prices = df_stats.max()
min_prices = df_stats.min()
summary_stats = pd.DataFrame({
    'Mean': mean_prices,
    'StdDev': std_prices,
    'Max': max_prices,
    'Min': min_prices
})

# Portfolio data
portfolio_stocks = ['INFY.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'TCS.NS', 'ICICIBANK.NS']
starting_capital = 1000000
weights = [0.25, 0.25, 0.20, 0.15, 0.15]
portfolio_data = df_nifty[['Date'] + portfolio_stocks].copy()
for stock in portfolio_stocks:
    portfolio_data[f'{stock}_return'] = portfolio_data[stock].pct_change()
return_cols = [f'{stock}_return' for stock in portfolio_stocks]
portfolio_data['portfolio_return'] = portfolio_data[return_cols].dot(weights)
portfolio_data['portfolio_value'] = starting_capital * (1 + portfolio_data['portfolio_return']).cumprod()
portfolio_mean_return = portfolio_data['portfolio_return'].mean()
portfolio_std_return = portfolio_data['portfolio_return'].std()
final_value = portfolio_data['portfolio_value'].iloc[-1]

# --- Sections ---

if section == "Dataset Overview":
    st.subheader("Dataset Preview (First 5 Rows)")
    st.dataframe(df_nifty.head())
    st.markdown("""
    After loading the dataset, we noticed there were 24 trading days (rows) and 51 columns (50 stocks + Date).  
    The only missing values are in the HDFC.NS column, which has 24 missing entries — meaning it's entirely missing for this period.
    """)

elif section == "Data Shape":
    st.subheader("Dataset Shape")
    st.write(f"Rows: {df_nifty.shape[0]}")
    st.write(f"Columns: {df_nifty.shape[1]}")

elif section == "Missing Values":
    st.subheader("Missing Values")
    missing = df_nifty.isnull().sum()
    st.dataframe(missing[missing > 0] if missing.sum() > 0 else "✅ No missing values!")

elif section == "Descriptive Stats":
    st.subheader("Descriptive Statistics")
    st.dataframe(summary_stats)

    # Histogram of mean prices
    st.markdown("#### Histogram of Mean Closing Prices")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    sns.histplot(mean_prices, bins=15, kde=True, ax=ax1)
    ax1.set_title('Mean Closing Prices')
    ax1.set_xlabel('Mean Price')
    ax1.set_ylabel('Frequency')
    st.pyplot(fig1)

    st.markdown("""
    The histogram above shows the distribution of mean closing prices for all NIFTY50 stocks during the analyzed period.  
    Most stocks have mean prices clustered between ₹943 and ₹3,655, representing the interquartile range (middle 50%) of the distribution.  
    The distribution is right-skewed, with a few high-priced outliers exceeding ₹20,000.  
    This skewness suggests the presence of extreme values that may disproportionately impact index performance or portfolio weighting.
    """)

    # Volatility bar chart
    st.markdown("#### Volatility (Standard Deviation) per Stock")
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    std_sorted = std_prices.sort_values(ascending=False)
    sns.barplot(x=std_sorted.index, y=std_sorted.values, ax=ax2)
    ax2.set_title('Volatility of NIFTY50 Stocks')
    ax2.set_ylabel('Std. Deviation')
    ax2.set_xlabel('Stock')
    ax2.tick_params(axis='x', rotation=90)
    st.pyplot(fig2)

    st.markdown("""
    The descriptive statistics show SHREECEM has the highest mean price (₹25299.9) while TATASTEEL has the lowest (₹152.2).  
    The volatility chart reveals BAJAJ-AUTO.NS is the most volatile stock, followed by SHREECEM.NS and BAJFINANCE.NS.
    """)

    # Highlight high/low mean return
    st.markdown("### Insights")
    st.success(f"📈 **Highest Mean Return:** {summary_stats['Mean'].idxmax()} — {summary_stats['Mean'].max():.4f}")
    st.error(f"📉 **Lowest Mean Return:** {summary_stats['Mean'].idxmin()} — {summary_stats['Mean'].min():.4f}")

    st.markdown("#### 🔝 Top 5 Stocks by Mean Return")
    st.dataframe(summary_stats.sort_values('Mean', ascending=False).head())

    st.markdown("#### 🔻 Bottom 5 Stocks by Mean Return")
    st.dataframe(summary_stats.sort_values('Mean', ascending=True).head())

elif section == "Portfolio Simulation":
    st.subheader("💼 Portfolio Simulation")
    st.markdown("Portfolio consists of: `INFY.NS`, `RELIANCE.NS`, `HDFCBANK.NS`, `TCS.NS`, `ICICIBANK.NS`")
    st.write(f"**Weights**: {weights}")
    st.write(f"**Mean Daily Return:** `{portfolio_mean_return:.4f}`")
    st.write(f"**Standard Deviation:** `{portfolio_std_return:.4f}`")
    st.write(f"**Final Portfolio Value:** ₹{final_value:,.2f}")

    st.markdown("#### 📈 Portfolio Value Over Time")
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(portfolio_data['Date'], portfolio_data['portfolio_value'], color='blue')
    ax3.set_title("Portfolio Value Over Time")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Portfolio Value (₹)")
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(alpha=0.3)
    st.pyplot(fig3)

    # --- Risk Analysis: Value at Risk (VaR) ---
    st.subheader("==== Value at Risk (VaR) - 95% Confidence Level ====")
    st.markdown("### Individual Stock VaR (1-day, historical method):")

    individual_returns = {}
    individual_var = {}

    for stock in portfolio_stocks:
        if stock in df_nifty.columns:
            returns = df_nifty[stock].pct_change().dropna()
            individual_returns[stock] = returns
            var_95 = returns.quantile(0.05)
            individual_var[stock] = var_95
            st.write(f"**{stock}**: VaR = `{var_95:.4f}` ({var_95*100:.2f}%)")

    st.info("Note: Negative VaR means a potential daily loss with 95% confidence.")
    st.markdown("""
    The chart below shows the distribution of daily returns for the portfolio, with the red dashed line marking the 1-day 95% Value at Risk (VaR).  
    This means that, with 95% confidence, the portfolio is not expected to lose more than about ₹8,500 in a single day.
    """)

    portfolio_returns = portfolio_data['portfolio_return'].dropna()
    VaR_95_return = round(np.percentile(portfolio_returns, 5), 4)
    VaR_95_rupees = VaR_95_return * starting_capital

    st.markdown("### Portfolio VaR (1-day, historical method):")
    st.write(f"Portfolio VaR = `{VaR_95_return:.4f}` ({VaR_95_return*100:.2f}%)")
    st.write(f"Estimated 1-day potential loss: ₹ `{abs(VaR_95_rupees):,.2f}`")

    fig4, ax4 = plt.subplots(figsize=(10, 5))
    sns.histplot(portfolio_returns, bins=15, kde=True, color='skyblue', ax=ax4)
    ax4.axvline(VaR_95_return, color='red', linestyle='--', linewidth=2, label='95% VaR')
    ax4.set_title('Portfolio Daily Returns Distribution with 95% VaR')
    ax4.set_xlabel('Daily Return')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    st.pyplot(fig4)

    # --- Technical Analysis: Moving Averages ---
    st.subheader("📈 Technical Analysis: Moving Averages")
    stocks_for_ta = ['TCS.NS', 'RELIANCE.NS']

    fig5, axes = plt.subplots(2, 1, figsize=(12, 10))  # 2 rows, 1 col

    for i, stock in enumerate(stocks_for_ta):
        if stock in df_nifty.columns:
            ma_7 = df_nifty[stock].rolling(window=7).mean()
            ma_14 = df_nifty[stock].rolling(window=14).mean()

            axes[i].plot(df_nifty['Date'], df_nifty[stock], label=f'{stock} Price', color='blue')
            axes[i].plot(df_nifty['Date'], ma_7, label='7-day MA', color='orange', linestyle='--')
            axes[i].plot(df_nifty['Date'], ma_14, label='14-day MA', color='red', linestyle='-.')
            axes[i].set_title(f'{stock} - Price with 7-day and 14-day Moving Averages')
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('Price (₹)')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig5)


    # --- Technical Analysis: Support and Resistance ---
    st.subheader("📊 Technical Analysis: Support and Resistance")

    TCS = df_nifty['TCS.NS']
    RELIANCE = df_nifty['RELIANCE.NS']

    tcs_ma7 = TCS.rolling(window=7).mean()
    tcs_ma14 = TCS.rolling(window=14).mean()
    reliance_ma7 = RELIANCE.rolling(window=7).mean()
    reliance_ma14 = RELIANCE.rolling(window=14).mean()

    fig6, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 cols for side-by-side

    # TCS Support/Resistance plot
    axes[0].plot(TCS.index, TCS.values, label='TCS Price', color='blue')
    axes[0].plot(tcs_ma7.index, tcs_ma7.values, label='7-day MA', color='orange')
    axes[0].plot(tcs_ma14.index, tcs_ma14.values, label='14-day MA', color='green')
    axes[0].axhline(y=4285, color='red', linestyle='--', label='Support (~₹4285)')
    axes[0].axhline(y=4427, color='purple', linestyle='--', label='Resistance (7d MA)')
    axes[0].axhline(y=4453, color='brown', linestyle='--', label='Resistance (14d MA)')
    axes[0].set_title('TCS: Price, MA, Support/Resistance')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Price (₹)')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # RELIANCE Support/Resistance plot
    axes[1].plot(RELIANCE.index, RELIANCE.values, label='RELIANCE Price', color='blue')
    axes[1].plot(reliance_ma7.index, reliance_ma7.values, label='7-day MA', color='orange')
    axes[1].plot(reliance_ma14.index, reliance_ma14.values, label='14-day MA', color='green')
    axes[1].axhline(y=2972, color='red', linestyle='--', label='Resistance (~₹2972)')
    axes[1].axhline(y=2947, color='purple', linestyle='--', label='Support (7d MA)')
    axes[1].axhline(y=2953, color='brown', linestyle='--', label='Support (14d MA)')
    axes[1].set_title('RELIANCE: Price, MA, Support/Resistance')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Price (₹)')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig6)

    st.success('Charts plotted with moving averages and support/resistance levels separately.')


    st.markdown("""
    The moving average (MA) indicators help smooth out price action by filtering out short-term fluctuations.  
    We have plotted the 7-day and 14-day MAs for TCS.NS and RELIANCE.NS.

    - If the short-term MA (7-day) crosses above the long-term MA (14-day), it's often seen as a **bullish signal**.
    - If it crosses below, it might suggest a **bearish trend**.
    
    Recommendations:

    1. Keep reviewing your strategy — this approach earned a small gain with low volatility.
    2. Watch RELIANCE — it may continue to perform well.
    3. Consider adjusting your TCS holdings if weakness persists.
    4. Be careful with very volatile stocks — they're better for short-term opportunities.
    5. Use stop-loss orders and protective strategies to manage risk.
    6. Rebalance your portfolio regularly to stay aligned with your risk profile and market shifts.
    """)
