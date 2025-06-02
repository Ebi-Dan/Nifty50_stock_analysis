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

# Remove 'HDFC.NS' and prepare stock list
stocks = [col for col in df_nifty.columns if col != 'Date']
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
    portfolio_data[f'{stock}_return'] = portfolio_data[stock].pct_change() * 100
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
    st.markdown("""##
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

    st.markdown("""##
    The histogram above shows the distribution of mean closing prices for all NIFTY50 stocks during the analyzed period.
    Most stocks have mean prices clustered between ₹943 and ₹3,655, representing the interquartile range (middle 50%) 
    of the distribution. The distribution is right-skewed, with a few high-priced outliers exceeding ₹20,000. 
    This skewness suggests the presence of extreme values that may disproportionately impact index performance or 
    portfolio weighting.
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

    st.markdown("""##
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


# Part 4A: Risk Analysis – Value at Risk (VaR) at 95% Confidence Level
portfolio_stocks = ['INFY.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'TCS.NS', 'ICICIBANK.NS']
individual_returns = {}
individual_var = {}

st.subheader("==== Value at Risk (VaR) - 95% Confidence Level ====")
st.markdown("### Individual Stock VaR (1-day, historical method):")

for stock in portfolio_stocks:
    if stock in df_nifty.columns:
        returns = df_nifty[stock].pct_change().dropna()
        individual_returns[stock] = returns
        var_95 = returns.quantile(0.05)
        individual_var[stock] = var_95
        st.write(f"**{stock}**: VaR = `{var_95:.4f}` ({var_95*100:.2f}%)")

st.info("Note: Negative VaR means a potential daily loss with 95% confidence.")
st.markdown("""###
The chart below shows the distribution of daily returns for the portfolio, with the red dashed line marking the 1-day 95% Value at Risk (VaR). 
This means that, with 95% confidence, the portfolio is not expected to lose more than about ₹8,500 in a single day."""
)
# Part 4B: Portfolio VaR
portfolio_returns = portfolio_data['portfolio_return'].dropna() / 100

VaR_95_return = round(np.percentile(portfolio_returns, 5), 4)
VaR_95_rupees = VaR_95_return * starting_capital

st.markdown("### Portfolio VaR (1-day, historical method):")
st.write(f"Portfolio VaR = `{VaR_95_return:.4f}` ({VaR_95_return*100:.2f}%)")
st.write(f"Estimated 1-day potential loss: ₹ `{abs(VaR_95_rupees):,.2f}`")

fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.histplot(portfolio_returns, bins=15, kde=True, color='skyblue', ax=ax1)
ax1.axvline(VaR_95_return, color='red', linestyle='--', linewidth=2, label='95% VaR')
ax1.set_title('Portfolio Daily Returns Distribution with 95% VaR')
ax1.set_xlabel('Daily Return')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# Part 5: Technical Analysis
stocks_for_ta = ['TCS.NS', 'RELIANCE.NS']
fig2, axes = plt.subplots(2, 1, figsize=(12, 10))

for i, stock in enumerate(stocks_for_ta):
    if stock in df_nifty.columns:
        ma_7 = df_nifty[stock].rolling(window=7).mean()
        ma_14 = df_nifty[stock].rolling(window=14).mean()
        
        axes[i].plot(df_nifty.index, df_nifty[stock], label=f'{stock} Price', color='blue')
        axes[i].plot(df_nifty.index, ma_7, label='7-day MA', color='orange', linestyle='--')
        axes[i].plot(df_nifty.index, ma_14, label='14-day MA', color='red', linestyle='-.')
        axes[i].set_title(f'{stock} - Price with 7-day and 14-day Moving Averages')
        axes[i].set_xlabel('Trading Days')
        axes[i].set_ylabel('Price (INR)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

        current_price = df_nifty[stock].iloc[-1]
        current_ma7 = ma_7.iloc[-1]
        current_ma14 = ma_14.iloc[-1]

        st.markdown(f"### {stock} Current Analysis")
        st.write(f"Current Price: ₹{current_price:.2f}")
        st.write(f"7-day MA: ₹{current_ma7:.2f}")
        st.write(f"14-day MA: ₹{current_ma14:.2f}")
        
        if current_price > current_ma7 > current_ma14:
            trend = "Strong Bullish"
        elif current_price > current_ma7 and current_price > current_ma14:
            trend = "Bullish"
        elif current_price < current_ma7 < current_ma14:
            trend = "Strong Bearish"
        elif current_price < current_ma7 and current_price < current_ma14:
            trend = "Bearish"
        else:
            trend = "Neutral/Mixed"

        st.write(f"**Trend**: {trend}")

st.pyplot(fig2)

# Price, MA & Support/Resistance chart
TCS = df_nifty['TCS.NS']
RELIANCE = df_nifty['RELIANCE.NS']
tcs_ma7 = TCS.rolling(window=7).mean()
tcs_ma14 = TCS.rolling(window=14).mean()
reliance_ma7 = RELIANCE.rolling(window=7).mean()
reliance_ma14 = RELIANCE.rolling(window=14).mean()

fig3 = plt.figure(figsize=(14, 6))

# TCS Plot
plt.subplot(1, 2, 1)
plt.plot(TCS.index, TCS.values, label='TCS Price', color='blue')
plt.plot(tcs_ma7.index, tcs_ma7.values, label='7-day MA', color='orange')
plt.plot(tcs_ma14.index, tcs_ma14.values, label='14-day MA', color='green')
plt.axhline(y=4285, color='red', linestyle='--', label='Support (~₹4285)')
plt.axhline(y=4427, color='purple', linestyle='--', label='Resistance (7d MA)')
plt.axhline(y=4453, color='brown', linestyle='--', label='Resistance (14d MA)')
plt.title('TCS: Price, MA, Support/Resistance')
plt.xlabel('Date')
plt.ylabel('Price (₹)')
plt.legend()

# RELIANCE Plot
plt.subplot(1, 2, 2)
plt.plot(RELIANCE.index, RELIANCE.values, label='RELIANCE Price', color='blue')
plt.plot(reliance_ma7.index, reliance_ma7.values, label='7-day MA', color='orange')
plt.plot(reliance_ma14.index, reliance_ma14.values, label='14-day MA', color='green')
plt.axhline(y=2972, color='red', linestyle='--', label='Resistance (~₹2972)')
plt.axhline(y=2947, color='purple', linestyle='--', label='Support (7d MA)')
plt.axhline(y=2953, color='brown', linestyle='--', label='Support (14d MA)')
plt.title('RELIANCE: Price, MA, Support/Resistance')
plt.xlabel('Date')
plt.ylabel('Price (₹)')
plt.legend()

plt.tight_layout()
st.pyplot(fig3)
st.success('Charts plotted with support and resistance levels for TCS and RELIANCE.')

# Part 6: Summary Report

# Find the three most volatile stocks (highest std dev of closing prices)
most_volatile = std_prices.sort_values(ascending=False).head(3)

report = f"""
INVESTING WITH NIFTY50: STRATEGY SUMMARY

DATA OVERVIEW:
- We looked at 24 trading days (Aug 20 – Sep 20, 2024)
- Checked 49 stocks (HDFC.NS was left out because data was missing)

TOP 3 MOST VOLATILE STOCKS (Biggest daily price changes):
1. {most_volatile.index[0]}: ₹{most_volatile.iloc[0]:.2f} average daily price movement
2. {most_volatile.index[1]}: ₹{most_volatile.iloc[1]:.2f} average daily price movement
3. {most_volatile.index[2]}: ₹{most_volatile.iloc[2]:.2f} average daily price movement

Key Points:

Compared to the most volatile stock (like {most_volatile.index[0]} with ₹{most_volatile.iloc[0]:.2f} daily swings), the portfolio had smaller price ups and downs but still lost a lot of money.

The stock with the smallest daily price change, TATASTEEL.NS, moves about ₹1.89 per day, which means it's safer but usually grows slower.

Even though the portfolio had many different stocks (diversified), it still lost a lot. This shows that spreading money around helps, but managing risk is also very important.

Technical Update:

    - TCS is trading below its short-term average prices → short-term outlook is weak.
    - RELIANCE is trading above its short-term average prices → short-term outlook is strong.

Portfolio Performance Compared to Individual Stocks:

- Average daily gain: 0.12%
- Daily ups and downs (volatility): 0.63%
- Total loss in value: ₹779,298.91
- Overall loss in 24 days: 177.93%

The top 3 most volatile stocks had much bigger daily price changes than the portfolio. These stocks can give higher quick profits but also bigger losses.

TATASTEEL.NS is the most stable stock here.

Diversifying helps lower risk, but how well the portfolio does depends on which stocks are chosen, how much money is put into each, and market conditions.

Recommendations:

1. Review your strategy — the current way lost a lot of money even with lower risk.
2. Watch RELIANCE — it may keep doing well.
3. Think about lowering your TCS holdings if the weak trend continues.
4. Be careful with very volatile stocks — better for short-term trades only.
5. Use stop-loss limits and think about ways to protect your investments.
6. Check and adjust your portfolio regularly to fit your risk comfort and market changes.
"""

# The top 3 most volatile stocks and report
st.markdown("### Summary Report")
st.dataframe(most_volatile)
st.markdown(report)