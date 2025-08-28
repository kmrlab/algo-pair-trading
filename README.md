## 🌟 Support This Project

Found this project useful? Please consider:

- ⭐ **Starring this repository** - It helps others discover the project
- 🍴 **Forking** and contributing improvements  
- 📢 **Sharing** with the trading and quantitative finance community
- 💡 **Opening issues** for bugs or feature requests
- 🚀 **Contributing** code, documentation, or examples

**Created with ❤️ by [Kristofer Meio-Renn](https://github.com/kmrlab)**

---

## How To Share

Spread the word about this project! Share with the trading and quantitative finance community:

- 📘 [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://github.com/kmrlab/algo-pair-trading)
- 💼 [Share on LinkedIn](https://www.linkedin.com/sharing/share-offsite/?url=https://github.com/kmrlab/algo-pair-trading)
- 📱 [Share on Telegram](https://t.me/share/url?url=https://github.com/kmrlab/algo-pair-trading&text=Professional-grade%20pairs%20trading%20and%20statistical%20arbitrage%20algorithms%20for%20crypto%20trading)
- 🐦 [Share on X (Twitter)](https://twitter.com/intent/tweet?url=https://github.com/kmrlab/algo-pair-trading&text=Check%20out%20this%20advanced%20pairs%20trading%20system%20for%20crypto%20markets!%20%23QuantitativeFinance%20%23PairsTrading%20%23OpenSource)

---

# Pairs Trading & Statistical Arbitrage

[![Stars](https://img.shields.io/github/stars/kmrlab/algo-pair-trading?style=social)](https://github.com/kmrlab/algo-pair-trading)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)

---

## 📚 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [File Descriptions](#-file-descriptions)
  - [analysis_correlation.py](#analysis_correlationpy)
  - [analysis_cointegration.py](#analysis_cointegrationpy)
  - [analysis_half_life.py](#analysis_half_lifepy)
  - [data_download.py](#data_downloadpy)
  - [stat_arbitrage_strategy.py](#stat_arbitrage_strategypy)
  - [stat_arbitrage_backtest.py](#stat_arbitrage_backtestpy)
- [Usage Examples](#-usage-examples)
- [Mathematical Background](#-mathematical-background)
- [Strategy Performance](#-strategy-performance)
- [Data Requirements](#-data-requirements)
- [Contributing](#-contributing)
- [License](#-license)
- [Disclaimer](#-disclaimer)

## 🎯 Overview

This repository contains a comprehensive suite of pairs trading and statistical arbitrage algorithms designed for cryptocurrency markets. The system implements professional-grade quantitative methods used by funds and traders.

The project provides end-to-end workflow from data acquisition and analysis to strategy implementation and backtesting, focusing on identifying cointegrated pairs and exploiting mean-reverting spreads.

## ✨ Features

- **Data Acquisition**: Automated data download from Binance and Bybit APIs
- **Statistical Analysis**: Correlation, cointegration, and stationarity testing
- **Half-Life Analysis**: Mean reversion speed calculation
- **Strategy Implementation**: Professional pairs trading algorithms
- **Backtesting Engine**: Comprehensive performance testing with realistic costs
- **Risk Management**: Position sizing and drawdown controls
- **Visualization**: Professional charts and analytical reports
- **Multiple Exchanges**: Support for Binance and Bybit data
- **Configurable Parameters**: Flexible strategy customization

## 🚀 Installation

### Requirements

- Python 3.8+
- NumPy, Pandas, SciPy for numerical computing
- Statsmodels for econometric analysis
- Matplotlib, Seaborn, Plotly for visualization
- Exchange APIs (python-binance, pybit)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/kmrlab/algo-pair-trading.git
cd algo-pair-trading

# Install dependencies
pip install -r requirements.txt

# Run correlation analysis
python analysis_correlation.py

# Run cointegration analysis
python analysis_cointegration.py

# Run half-life analysis
python analysis_half_life.py
```

## 📁 Project Structure

```
algo-pair-trading/
├── analysis_correlation.py      # Correlation analysis between assets
├── analysis_cointegration.py    # Cointegration testing (Engle-Granger)
├── analysis_half_life.py        # Mean reversion speed analysis
├── data_download.py             # Data acquisition from Bybit API
├── stat_arbitrage_strategy.py   # Main trading strategy analyzer
├── stat_arbitrage_backtest.py   # Backtesting engine
├── bybit_data/                  # Historical price data
│   ├── BTCUSDT_5m_*.csv
│   ├── ETHUSDT_5m_*.csv
│   └── XRPUSDT_5m_*.csv
├── correlation/                 # Correlation analysis results
├── cointegration/               # Cointegration test results
├── half_life/                   # Half-life analysis results
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 📋 File Descriptions

### `analysis_correlation.py`

**Advanced Correlation Analysis Engine**

Comprehensive correlation analysis system for identifying strongly correlated asset pairs:

- **Pearson Correlation**: Static correlation calculation between asset returns
- **Rolling Correlation**: Time-varying correlation analysis (1d, 7d, 30d windows)
- **Correlation Stability**: Assessment of correlation consistency over time
- **Statistical Filtering**: Z-score based outlier removal and winsorization
- **Visualization**: Professional heatmaps and correlation matrices

**Key Features:**
- Handles extreme price movements and outliers
- Multi-timeframe correlation analysis
- Correlation stability scoring
- Professional visualization with dark theme
- Comprehensive text reports with trading recommendations

**Use Case**: First step in pairs selection - identify assets with strong correlation suitable for pairs trading.

### `analysis_cointegration.py`

**Engle-Granger Cointegration Testing**

Professional cointegration analysis using the two-step Engle-Granger procedure:

- **Cointegration Testing**: Identifies long-term equilibrium relationships
- **ADF Stationarity Tests**: Validates spread stationarity
- **Hedge Ratio Calculation**: Optimal position sizing ratios
- **Cointegration Scoring**: Composite metric for pair ranking
- **Ljung-Box Testing**: Residual autocorrelation diagnostics

**Key Features:**
- Bidirectional testing (both Y~X and X~Y relationships)
- Automated optimal lag selection
- R² goodness-of-fit assessment
- Professional interpretation and recommendations
- Multi-asset pairwise analysis

**Use Case**: Validate that correlated pairs have genuine cointegration relationships for statistical arbitrage.

### `analysis_half_life.py`

**Mean Reversion Speed Analysis**

Half-life analysis for determining optimal trading parameters:

- **AR(1) Model**: Δspread(t) = α + λ×spread(t-1) + ε(t)
- **Half-Life Calculation**: -ln(2) / ln(1 + λ)
- **Rolling Analysis**: Time-varying half-life estimation
- **Stationarity Validation**: ADF testing for each window
- **Trading Recommendations**: Parameter optimization guidance

**Key Features:**
- Configurable rolling windows (default: 500 observations)
- Statistical filtering of valid estimates
- Conversion to trading timeframes (hours/minutes)
- Strategy parameter recommendations
- Professional visualization of results

**Use Case**: Determine how quickly spreads revert to mean, crucial for position holding times and exit strategies.

### `data_download.py`

**Multi-Exchange Data Acquisition**

Professional data downloader with robust API handling:

- **Bybit API Integration**: Unified Trading API for perpetual futures
- **Rate Limiting**: Respects exchange API limits
- **Retry Logic**: Handles network errors and timeouts
- **Batch Processing**: Efficient bulk data retrieval
- **Data Validation**: Quality checks and formatting

**Key Features:**
- Support for 15+ major cryptocurrencies
- 5-minute OHLCV data acquisition
- Automatic timestamp synchronization
- Progress tracking with tqdm
- Configurable date ranges and symbols

**Use Case**: Acquire high-quality historical data required for all statistical analyses.

### `stat_arbitrage_strategy.py`

**Main Strategy Analysis Engine**

Comprehensive pairs trading strategy analyzer:

- **Pair Selection**: Automated correlation and cointegration screening
- **Signal Generation**: Z-score based entry/exit signals
- **Rolling Beta**: Dynamic hedge ratio calculation
- **Risk Assessment**: Statistical metrics and pair scoring
- **Interactive Visualization**: Plotly-based professional charts

**Key Features:**
- Multi-asset screening (24 symbols by default)
- Configurable selection criteria (correlation > 0.85, p-value < 0.05)
- Rolling correlation and beta analysis
- Z-score signal generation (entry: ±2.0, exit: ±0.5)
- Professional trading recommendations

**Use Case**: End-to-end strategy analysis from pair selection to signal generation.

### `stat_arbitrage_backtest.py`

**Professional Backtesting Engine**

Realistic backtesting with institutional-grade features:

- **Realistic Execution**: Transaction costs and slippage modeling
- **Position Management**: Leverage and capital allocation
- **Performance Metrics**: Sharpe ratio, max drawdown, win rate
- **Risk Controls**: Minimum holding periods and position limits
- **P&L Attribution**: Detailed trade-by-trade analysis

**Key Features:**
- Configurable initial capital and leverage
- Commission modeling (0.02% default)
- Minimum holding time constraints (4 periods = 20 minutes)
- Professional performance visualization
- Comprehensive trading statistics

**Use Case**: Validate strategy performance with realistic market conditions and costs.

## 💻 Usage Examples

### Basic Correlation Analysis

```python
from analysis_correlation import CorrelationAnalyzer

# Initialize analyzer
analyzer = CorrelationAnalyzer(data_dir='bybit_data', output_dir='correlation')

# Run full analysis
analyzer.run_full_analysis()

# Results:
# - correlation_heatmap_TIMESTAMP.png
# - correlation_report_TIMESTAMP.txt
```

### Cointegration Testing

```python
from analysis_cointegration import CointegrationAnalyzer

# Initialize analyzer
analyzer = CointegrationAnalyzer(data_dir='bybit_data', output_dir='cointegration')

# Run analysis
analyzer.run_full_analysis()

# Results:
# - Cointegrated pairs identification
# - Hedge ratios calculation
# - Professional reports and visualizations
```

### Half-Life Analysis

```python
from analysis_half_life import HalfLifeAnalyzer

# Initialize with custom rolling window
analyzer = HalfLifeAnalyzer(rolling_window=500)

# Run analysis
analyzer.run_analysis()

# Results:
# - Half-life estimates in hours
# - Trading parameter recommendations
# - Statistical validation
```

### Strategy Analysis

```python
from stat_arbitrage_strategy import PairsTradingAnalyzer

# Initialize analyzer
analyzer = PairsTradingAnalyzer()

# Run comprehensive analysis
analyzer.run_full_analysis()

# Results:
# - Top correlated pairs
# - Cointegration testing
# - Trading signals
# - Interactive visualizations
```

### Backtesting

```python
from stat_arbitrage_backtest import PairsTradingStrategy

# Initialize strategy
strategy = PairsTradingStrategy()

# Run backtest
strategy.run_backtest()

# Results:
# - Portfolio performance metrics
# - Trade-by-trade analysis
# - Professional performance charts
```

## 📊 Mathematical Background

### Cointegration Testing

**Engle-Granger Two-Step Procedure:**

1. **Cointegration Regression:**
   ```
   Y(t) = α + β × X(t) + ε(t)
   ```

2. **Stationarity Test:**
   ```
   ADF Test on residuals ε(t)
   H₀: ε(t) has unit root (not cointegrated)
   H₁: ε(t) is stationary (cointegrated)
   ```

### Half-Life Calculation

**AR(1) Model for Spread:**
```
Δspread(t) = α + λ × spread(t-1) + ε(t)
```

**Half-Life Formula:**
```
Half-Life = -ln(2) / ln(1 + λ)
```

Where λ < 0 indicates mean reversion.

### Z-Score Signal Generation

**Spread Calculation:**
```
Spread(t) = ln(P₁(t)) - β(t) × ln(P₂(t))
```

**Z-Score:**
```
Z(t) = (Spread(t) - μ(t)) / σ(t)
```

**Trading Signals:**
- Long Entry: Z < -2.0
- Short Entry: Z > +2.0
- Exit: |Z| < 0.5

### Performance Metrics

**Sharpe Ratio:**
```
Sharpe = (μ_returns - r_f) / σ_returns
```

**Maximum Drawdown:**
```
MDD = max(Peak - Trough) / Peak
```

## 📈 Strategy Performance

### Typical Performance Characteristics

- **Sharpe Ratio**: 1.5 - 3.0 (depending on market conditions)
- **Maximum Drawdown**: 5% - 15%
- **Win Rate**: 60% - 75%
- **Average Trade Duration**: 2-8 hours
- **Correlation Requirement**: > 0.85
- **Half-Life**: 4-24 hours optimal

### Market Conditions

**Ideal Conditions:**
- High correlation (> 0.9)
- Strong cointegration (p < 0.01)
- Short half-life (< 12 hours)
- Stable volatility environment

**Challenging Conditions:**
- Market regime changes
- Correlation breakdown
- High volatility periods
- Low liquidity environments

## 💾 Data Requirements

### Historical Data Specifications

- **Timeframe**: 5-minute OHLCV candles
- **History**: Minimum 30 days, optimal 90+ days
- **Exchanges**: Binance, Bybit perpetual futures
- **Assets**: Major cryptocurrencies (BTC, ETH, etc.)
- **Quality**: No missing data, validated timestamps

### Data Sources

1. **Bybit API**: Primary data source via `data_download.py`
2. **Binance API**: Alternative via `stat_arbitrage_*.py` files
3. **Custom Data**: CSV format with standard OHLCV columns

### Storage Structure

```
bybit_data/
├── BTCUSDT_5m_2024-01-01_to_2025-08-10.csv
├── ETHUSDT_5m_2024-01-01_to_2025-08-10.csv
└── XRPUSDT_5m_2024-01-01_to_2025-08-10.csv
```

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution

- **Additional Exchanges**: Implement other exchange APIs
- **Advanced Models**: Kalman filters, VECM models
- **Risk Management**: Dynamic position sizing, VaR models
- **Machine Learning**: ML-based pair selection
- **Real-time Trading**: Live strategy execution
- **Alternative Assets**: Stocks, commodities, forex pairs

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

**IMPORTANT RISK DISCLOSURE**

This software is provided for educational and research purposes only. The algorithms and methodologies contained herein are not investment advice and should not be construed as such.

### Key Risk Factors:

- **Market Risk**: Cryptocurrency markets are extremely volatile
- **Model Risk**: Statistical relationships may break down
- **Execution Risk**: Slippage and transaction costs impact performance
- **Liquidity Risk**: Low liquidity can prevent strategy execution
- **Technology Risk**: API failures, network issues, software bugs
- **Regulatory Risk**: Changing regulations may affect trading

### Professional Disclaimer:

- **No Investment Advice**: This is not personalized investment advice
- **Backtesting Limitations**: Historical performance doesn't guarantee future results
- **Due Diligence Required**: Conduct thorough testing before live trading
- **Professional Consultation**: Consult qualified financial advisors
- **Risk Management**: Never risk more capital than you can afford to lose

### Strategy-Specific Risks:

- **Correlation Breakdown**: Pairs may decorrelate during market stress
- **Mean Reversion Failure**: Spreads may trend instead of reverting
- **Regime Changes**: Market structure changes can invalidate models
- **Execution Timing**: High-frequency nature requires low-latency execution
- **Capital Requirements**: Requires sufficient capital for pair positions

### Liability Limitation:

The authors and contributors shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use of this software, including but not limited to financial losses, trading losses, or missed opportunities.

**USE AT YOUR OWN RISK**

By using this software, you acknowledge that you understand these risks and agree to use the software solely at your own discretion and risk.

---

**Happy Trading! 📈**
