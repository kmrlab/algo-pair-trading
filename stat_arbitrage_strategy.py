"""
██╗  ██╗███╗   ███╗██████╗ ██╗      █████╗ ██████╗ 
██║ ██╔╝████╗ ████║██╔══██╗██║     ██╔══██╗██╔══██╗
█████╔╝ ██╔████╔██║██████╔╝██║     ███████║██████╔╝
██╔═██╗ ██║╚██╔╝██║██╔══██╗██║     ██╔══██║██╔══██╗
██║  ██╗██║ ╚═╝ ██║██║  ██║███████╗██║  ██║██████╔╝
╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝ 

Crafted with ❤️ by Kristofer Meio-Renn

Found this useful? Star the repo to show your support! Thank you!
GitHub: https://github.com/kmrlab
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os

from binance.client import Client
from datetime import datetime, timedelta
import time
from itertools import combinations
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# ANALYSIS SETTINGS
# =============================================================================

# Time parameters
START_DATE = '2024-01-01'          # Analysis start date
END_DATE = '2024-01-15'            # Analysis end date
TIMEFRAME = '5m'                   # Data timeframe

# Asset list for analysis (futures symbols)
SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARBUSDT', 
    'SEIUSDT', 'SUIUSDT', 'DOGEUSDT', 'ADAUSDT', 'AAVEUSDT',
    'LINKUSDT', 'OPUSDT', 'ENAUSDT', 'AVAXUSDT', 'BCHUSDT',
    'NEARUSDT', 'LTCUSDT', 'GALAUSDT', 'TIAUSDT', 'UNIUSDT',
    'WLDUSDT', 'INJUSDT', 'DOTUSDT', 'APTUSDT'
]

# Pair selection criteria
TOP_PAIRS_COUNT = 20               # Number of top pairs by correlation
MIN_CORRELATION = 0.85             # Minimum correlation for selection
COINTEGRATION_P_VALUE = 0.05       # P-value for cointegration tests

# Stationarity parameters
ADF_P_VALUE = 0.05                 # P-value for ADF stationarity test

# Rolling correlation parameters
ROLLING_WINDOW = 96                # Window in periods (96 = 8 hours on 5-minute)

# Rolling beta and Z-Score parameters
LOOKBACK_PERIOD = 96               # Period for calculating rolling beta and Z-Score

# Trading signal parameters
ENTRY_THRESHOLD = 2.0              # Z-Score for position entry
EXIT_THRESHOLD = 0.5               # Z-Score for position exit
MIN_HOLDING_PERIODS = 4            # Minimum position holding time (4 periods = 20 minutes on 5m)

# Binance API settings
# Set your API credentials as environment variables or replace with your keys
API_KEY = os.getenv('BINANCE_API_KEY', 'your_binance_api_key_here')
API_SECRET = os.getenv('BINANCE_API_SECRET', 'your_binance_api_secret_here')

# API parameters
API_DELAY = 1.0                    # Delay between API requests (seconds)
KLINES_LIMIT = 1000                # Maximum candles per request

# =============================================================================

class PairsTradingAnalyzer:
    def __init__(self):
        """
        Initialize pairs trading analyzer
        """
        # Binance client with API keys
        self.client = Client(API_KEY, API_SECRET)
        
        # Using settings from configuration
        self.symbols = SYMBOLS
        self.data = {}
        self.prices_df = None
        self.returns_df = None
        
    def fetch_futures_data(self, symbol, start_date=START_DATE, end_date=END_DATE, interval=TIMEFRAME):
        """
        Fetch futures data from Binance considering API limits
        """
        try:
            print(f"Fetching data for {symbol}...")
            
            # Convert dates to milliseconds
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
            
            all_klines = []
            current_start = start_ts
            
            # Fetch data in chunks considering 1000 candle limit
            while current_start < end_ts:
                print(f"  Loading data from {datetime.fromtimestamp(current_start/1000)}")
                
                # Get candlestick data
                klines = self.client.futures_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_str=current_start,
                    end_str=end_ts,
                    limit=KLINES_LIMIT
                )
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                
                # Update starting point for next request
                # Take last candle time + 1 interval
                last_timestamp = int(klines[-1][0])
                
                # Calculate interval in milliseconds
                interval_ms = self._get_interval_ms(interval)
                current_start = last_timestamp + interval_ms
                
                # Pause between requests
                time.sleep(API_DELAY)
                
                # Check if we reached end date
                if current_start >= end_ts:
                    break
            
            if not all_klines:
                print(f"No data for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(all_klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Process data
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert to numeric values
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            # Remove duplicates by index (time)
            df = df[~df.index.duplicated(keep='first')]
            
            # Sort by time
            df = df.sort_index()
            
            print(f"  Loaded {len(df)} candles for {symbol}")
            print(f"  Period: {df.index[0]} - {df.index[-1]}")
            
            return df[['close']]
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _get_interval_ms(self, interval):
        """
        Convert interval to milliseconds
        """
        interval_map = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
            '1M': 30 * 24 * 60 * 60 * 1000
        }
        return interval_map.get(interval, 15 * 60 * 1000)  # Default 15 minutes
    
    def load_all_data(self, start_date=START_DATE, end_date=END_DATE):
        """
        Load data for all symbols
        """
        print("Starting data loading...")
        print(f"Loading period: {start_date} - {end_date}")
        print(f"Timeframe: {TIMEFRAME}")
        print(f"Number of symbols: {len(self.symbols)}")
        
        successful_downloads = 0
        
        for i, symbol in enumerate(self.symbols, 1):
            print(f"\n[{i}/{len(self.symbols)}] Processing {symbol}")
            df = self.fetch_futures_data(symbol, start_date, end_date)
            if df is not None:
                self.data[symbol] = df
                successful_downloads += 1
            else:
                print(f"Failed to load data for {symbol}")
            
            # Progress
            print(f"Progress: {i}/{len(self.symbols)} ({i/len(self.symbols)*100:.1f}%)")
            
        print(f"\nSuccessfully loaded data for {successful_downloads} out of {len(self.symbols)} symbols")
        
        # Create common DataFrame with closing prices
        if self.data:
            self.prices_df = pd.DataFrame({
                symbol: data['close'] for symbol, data in self.data.items()
            })
            
            # Remove NaN rows
            self.prices_df.dropna(inplace=True)
            
            # Calculate returns
            self.returns_df = self.prices_df.pct_change().dropna()
            
            print(f"Data loaded for {len(self.data)} assets")
            print(f"Period: {self.prices_df.index[0]} - {self.prices_df.index[-1]}")
            print(f"Number of observations: {len(self.prices_df)}")
        else:
            print("Failed to load data")
    
    def calculate_correlation_analysis(self):
        """
        1. Correlation analysis between asset pairs
        """
        print("\n=== 1. CORRELATION ANALYSIS ===")
        
        # 1.1 Pearson correlation
        correlation_matrix = self.prices_df.corr(method='pearson')
        
        # Create heatmap
        plt.figure(figsize=(16, 14))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.3f', 
                   cmap='RdYlBu_r', center=0, square=True, cbar_kws={"shrink": .8})
        plt.title('Asset Correlation Matrix (Pearson)', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Top-20 pairs with highest correlation
        pairs_correlation = []
        symbols = list(correlation_matrix.columns)
        
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                pair = (symbols[i], symbols[j])
                corr_value = correlation_matrix.iloc[i, j]
                pairs_correlation.append({
                    'Pair': f"{pair[0]} - {pair[1]}",
                    'Asset1': pair[0],
                    'Asset2': pair[1],
                    'Correlation': corr_value
                })
        
        pairs_df = pd.DataFrame(pairs_correlation)
        top_20_pairs = pairs_df.nlargest(TOP_PAIRS_COUNT, 'Correlation')
        
        print(f"\nTop-{TOP_PAIRS_COUNT} pairs with highest correlation:")
        print(top_20_pairs.to_string(index=False))
        
        # Calculate correlation stability for top pairs
        correlation_stability = []
        
        for _, row in top_20_pairs.iterrows():
            asset1, asset2 = row['Asset1'], row['Asset2']
            
            # Rolling correlation
            rolling_corr = self.prices_df[asset1].rolling(ROLLING_WINDOW).corr(self.prices_df[asset2])
            rolling_corr = rolling_corr.dropna()
            
            if len(rolling_corr) > 0:
                corr_std = rolling_corr.std()
                corr_mean = rolling_corr.mean()
                stability_score = corr_mean / corr_std if corr_std != 0 else 0
                
                correlation_stability.append({
                    'Pair': row['Pair'],
                    'Asset1': asset1,
                    'Asset2': asset2,
                    'Static_Correlation': row['Correlation'],
                    'Rolling_Corr_Mean': corr_mean,
                    'Rolling_Corr_Std': corr_std,
                    'Stability_Score': stability_score
                })
        
        stability_df = pd.DataFrame(correlation_stability)
        
        return top_20_pairs, stability_df
    
    def cointegration_tests(self, pairs_df):
        """
        2. Cointegration tests
        """
        print("\n=== 2. COINTEGRATION TESTS ===")
        
        # Engle-Granger test
        print("\nEngle-Granger Test")
        
        engle_granger_results = []
        
        for _, row in pairs_df.iterrows():
            asset1, asset2 = row['Asset1'], row['Asset2']
            
            try:
                # Cointegration test
                score, p_value, _ = coint(self.prices_df[asset1], self.prices_df[asset2])
                
                is_cointegrated = p_value < COINTEGRATION_P_VALUE
                
                engle_granger_results.append({
                    'Pair': row['Pair'],
                    'Asset1': asset1,
                    'Asset2': asset2,
                    'EG_Score': score,
                    'EG_P_Value': p_value,
                    'Is_Cointegrated_EG': is_cointegrated
                })
                
            except Exception as e:
                print(f"Error in EG test for {row['Pair']}: {e}")
        
        eg_df = pd.DataFrame(engle_granger_results)
        cointegrated_eg = eg_df[eg_df['Is_Cointegrated_EG'] == True]
        
        print(f"\nPairs with cointegration by Engle-Granger test ({len(cointegrated_eg)} out of {len(eg_df)}):")
        if len(cointegrated_eg) > 0:
            print(cointegrated_eg[['Pair', 'EG_Score', 'EG_P_Value']].to_string(index=False))
        else:
            print("No cointegrated pairs found")
        
        return eg_df
    
    def filter_qualified_pairs(self, top_20_pairs, stability_df, eg_df):
        """
        Filter pairs by all criteria
        """
        print("\n=== PAIR FILTERING BY CRITERIA ===")
        
        # Selection criteria from settings:
        # 1. Correlation > MIN_CORRELATION
        # 2. In top pairs by correlation
        # 3. Cointegrated by EG test
        
        high_corr_pairs = top_20_pairs[top_20_pairs['Correlation'] > MIN_CORRELATION]
        
        # Merge all results
        qualified_pairs = high_corr_pairs.copy()
        
        # Add stability results
        qualified_pairs = qualified_pairs.merge(
            stability_df[['Pair', 'Static_Correlation', 'Rolling_Corr_Mean', 'Rolling_Corr_Std', 'Stability_Score']], 
            on='Pair', how='left'
        )
        
        # Add EG test results
        qualified_pairs = qualified_pairs.merge(
            eg_df[['Pair', 'EG_Score', 'EG_P_Value', 'Is_Cointegrated_EG']], 
            on='Pair', how='left'
        )
        
        # Final filtering
        final_pairs = qualified_pairs[
            (qualified_pairs['Correlation'] > MIN_CORRELATION) &
            (qualified_pairs['Is_Cointegrated_EG'] == True)
        ]
        
        print(f"\nPairs passing all selection criteria: {len(final_pairs)}")
        if len(final_pairs) > 0:
            print("Selection criteria:")
            print(f"- Correlation > {MIN_CORRELATION}")
            print(f"- Cointegration by Engle-Granger test (p-value < {COINTEGRATION_P_VALUE})")
            print(f"- In top-{TOP_PAIRS_COUNT} pairs by correlation")
            print("\nSelected pairs:")
            print(final_pairs[['Pair', 'Asset1', 'Asset2']].to_string(index=False))
        else:
            print("No pairs meeting all criteria")
            print(f"Using pairs with correlation > {MIN_CORRELATION} for methodology demonstration:")
            print("Criteria for demonstration:")
            print(f"- Correlation > {MIN_CORRELATION}")
            print(f"- In top-{TOP_PAIRS_COUNT} pairs by correlation")
            final_pairs = high_corr_pairs.head(5)  # Take top-5 for demonstration
            print("\nPairs for demonstration:")
            print(final_pairs[['Pair', 'Asset1', 'Asset2']].to_string(index=False))
        
        return final_pairs
    
    def calculate_rolling_beta(self, asset1_prices, asset2_prices, lookback_period=LOOKBACK_PERIOD):
        """
        Calculate rolling beta using linear regression
        """
        betas = []
        alphas = []
        
        for i in range(lookback_period, len(asset1_prices)):
            # Take data window
            y = asset1_prices.iloc[i-lookback_period:i].values
            X = asset2_prices.iloc[i-lookback_period:i].values.reshape(-1, 1)
            
            # Linear regression
            reg = LinearRegression().fit(X, y)
            betas.append(reg.coef_[0])
            alphas.append(reg.intercept_)
        
        # Create Series with correct index
        beta_series = pd.Series(betas, index=asset1_prices.index[lookback_period:])
        alpha_series = pd.Series(alphas, index=asset1_prices.index[lookback_period:])
        
        return beta_series, alpha_series
    
    def spread_analysis(self, qualified_pairs):
        """
        3. Spread analysis for selected pairs using rolling beta
        """
        print("\n=== 3. SPREAD ANALYSIS ===")
        print(f"Using rolling beta with lookback period: {LOOKBACK_PERIOD}")
        print(f"Minimum position holding time: {MIN_HOLDING_PERIODS} periods ({MIN_HOLDING_PERIODS * 5} minutes)")
        
        spread_results = []
        
        for _, row in qualified_pairs.iterrows():
            asset1, asset2 = row['Asset1'], row['Asset2']
            pair_name = row['Pair']
            
            print(f"\nAnalyzing pair: {pair_name}")
            
            # Output all available metrics for pair
            print(f"Static Correlation: {row.get('Static_Correlation', row['Correlation']):.6f}")
            print(f"Rolling Correlation Mean: {row.get('Rolling_Corr_Mean', 'N/A')}")
            print(f"Rolling Correlation Std: {row.get('Rolling_Corr_Std', 'N/A')}")
            print(f"EG Score: {row.get('EG_Score', 'N/A')}")
            print(f"EG P-Value: {row.get('EG_P_Value', 'N/A')}")
            
            # 3.1 Calculate rolling beta
            asset1_prices = self.prices_df[asset1]
            asset2_prices = self.prices_df[asset2]
            
            rolling_beta, rolling_alpha = self.calculate_rolling_beta(asset1_prices, asset2_prices, LOOKBACK_PERIOD)
            
            print(f"Rolling beta - mean: {rolling_beta.mean():.6f}, std: {rolling_beta.std():.6f}")
            print(f"Rolling alpha - mean: {rolling_alpha.mean():.6f}, std: {rolling_alpha.std():.6f}")
            
            # 3.2 Calculate spread with rolling beta
            # Align indices
            common_index = rolling_beta.index
            spread = asset1_prices.loc[common_index] - rolling_beta * asset2_prices.loc[common_index]
            
            # 3.3 Calculate rolling Z-Score
            spread_mean = spread.rolling(LOOKBACK_PERIOD).mean()
            spread_std = spread.rolling(LOOKBACK_PERIOD).std()
            spread_zscore = (spread - spread_mean) / spread_std
            
            # Remove NaN values
            spread_zscore = spread_zscore.dropna()
            spread = spread.loc[spread_zscore.index]
            
            # 3.4 Stationarity test (ADF) for spread
            adf_result = adfuller(spread.dropna())
            adf_statistic = adf_result[0]
            adf_p_value = adf_result[1]
            is_stationary = adf_p_value < ADF_P_VALUE
            
            print(f"ADF statistic: {adf_statistic:.6f}")
            print(f"ADF p-value: {adf_p_value:.6f}")
            print(f"Spread is stationary: {is_stationary}")
            
            # Save results
            spread_results.append({
                'Pair': pair_name,
                'Asset1': asset1,
                'Asset2': asset2,
                'Rolling_Beta_Mean': rolling_beta.mean(),
                'Rolling_Beta_Std': rolling_beta.std(),
                'Rolling_Alpha_Mean': rolling_alpha.mean(),
                'Rolling_Alpha_Std': rolling_alpha.std(),
                'ADF_Statistic': adf_statistic,
                'ADF_P_Value': adf_p_value,
                'Is_Stationary': is_stationary,
                'Static_Correlation': row.get('Static_Correlation', row['Correlation']),
                'Rolling_Corr_Mean': row.get('Rolling_Corr_Mean', None),
                'Rolling_Corr_Std': row.get('Rolling_Corr_Std', None),
                'EG_Score': row.get('EG_Score', None),
                'EG_P_Value': row.get('EG_P_Value', None),
                'Spread': spread,
                'Spread_ZScore': spread_zscore,
                'Rolling_Beta': rolling_beta,
                'Rolling_Alpha': rolling_alpha
            })
            
            # Visualization
            self.plot_spread_analysis_with_rolling_beta(
                spread, spread_zscore, rolling_beta,
                asset1_prices, asset2_prices,
                pair_name, asset1, asset2
            )
        
        return spread_results
    
    def plot_spread_analysis_with_rolling_beta(self, spread, spread_zscore, rolling_beta,
                                              asset1_prices, asset2_prices,
                                              pair_name, asset1, asset2):
        """
        Visualize spread analysis considering rolling beta and minimum holding time
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # 1. Two asset chart with entry/exit signals
        # Normalize prices for visual comparison
        common_index = spread_zscore.index
        price1_norm = (asset1_prices.loc[common_index] / asset1_prices.loc[common_index].iloc[0]) * 100
        price2_norm = (asset2_prices.loc[common_index] / asset2_prices.loc[common_index].iloc[0]) * 100
        
        ax1.plot(price1_norm.index, price1_norm, label=asset1, linewidth=1.5, alpha=0.8)
        ax1.plot(price2_norm.index, price2_norm, label=asset2, linewidth=1.5, alpha=0.8)
        
        # Generate signals based on Z-Score considering minimum holding time
        entry_threshold = ENTRY_THRESHOLD
        exit_threshold = EXIT_THRESHOLD
        min_holding = MIN_HOLDING_PERIODS
        
        # Find entry and exit points
        position = 0  # 0 - no position, 1 - long spread, -1 - short spread
        entry_points_long = []
        entry_points_short = []
        exit_points = []
        periods_in_position = 0
        
        for i in range(len(spread_zscore)):
            current_zscore = spread_zscore.iloc[i]
            
            if position == 0:  # No position
                if current_zscore > entry_threshold:  # Spread high - short spread
                    position = -1
                    periods_in_position = 0
                    entry_points_short.append((spread_zscore.index[i], price1_norm.iloc[i], price2_norm.iloc[i]))
                elif current_zscore < -entry_threshold:  # Spread low - long spread
                    position = 1
                    periods_in_position = 0
                    entry_points_long.append((spread_zscore.index[i], price1_norm.iloc[i], price2_norm.iloc[i]))
            
            elif position != 0:  # Has position
                periods_in_position += 1
                # Exit only after minimum holding time
                if periods_in_position >= min_holding and abs(current_zscore) < exit_threshold:
                    exit_points.append((spread_zscore.index[i], price1_norm.iloc[i], price2_norm.iloc[i]))
                    position = 0
                    periods_in_position = 0
        
        # Display signals on price chart
        for entry in entry_points_long:
            ax1.scatter(entry[0], entry[1], color='green', marker='^', s=100, alpha=0.8, 
                       label='Long Asset1' if entry == entry_points_long[0] else "")
            ax1.scatter(entry[0], entry[2], color='red', marker='v', s=100, alpha=0.8, 
                       label='Short Asset2' if entry == entry_points_long[0] else "")
        
        for entry in entry_points_short:
            ax1.scatter(entry[0], entry[1], color='red', marker='v', s=100, alpha=0.8, 
                       label='Short Asset1' if entry == entry_points_short[0] else "")
            ax1.scatter(entry[0], entry[2], color='green', marker='^', s=100, alpha=0.8, 
                       label='Long Asset2' if entry == entry_points_short[0] else "")
        
        for exit_point in exit_points:
            ax1.scatter(exit_point[0], exit_point[1], color='yellow', marker='o', s=80, alpha=0.8, 
                       edgecolors='black', label='Exit' if exit_point == exit_points[0] else "")
            ax1.scatter(exit_point[0], exit_point[2], color='yellow', marker='o', s=80, alpha=0.8, 
                       edgecolors='black')
        
        ax1.set_title(f'Asset Prices with Entry/Exit Signals: {pair_name}')
        ax1.set_ylabel('Normalized Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Rolling beta
        ax2.plot(rolling_beta.index, rolling_beta, linewidth=1.5, color='purple', label='Rolling Beta')
        ax2.axhline(y=rolling_beta.mean(), color='purple', linestyle='--', alpha=0.5, 
                   label=f'Mean: {rolling_beta.mean():.4f}')
        ax2.set_title(f'Rolling Beta (lookback={LOOKBACK_PERIOD})')
        ax2.set_ylabel('Beta')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Z-Score with entry/exit levels
        ax3.plot(spread_zscore.index, spread_zscore, linewidth=1, alpha=0.7, color='blue')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.axhline(y=entry_threshold, color='red', linestyle='--', alpha=0.7, 
                   label=f'Entry ±{entry_threshold}σ')
        ax3.axhline(y=-entry_threshold, color='red', linestyle='--', alpha=0.7)
        ax3.axhline(y=exit_threshold, color='orange', linestyle=':', alpha=0.7, 
                   label=f'Exit ±{exit_threshold}σ')
        ax3.axhline(y=-exit_threshold, color='orange', linestyle=':', alpha=0.7)
        
        # Display signals on Z-Score
        for entry in entry_points_long + entry_points_short:
            idx = spread_zscore.index.get_loc(entry[0])
            ax3.scatter(entry[0], spread_zscore.iloc[idx], 
                       color='green' if entry in entry_points_long else 'red', 
                       marker='^' if entry in entry_points_long else 'v', s=100, alpha=0.8)
        
        for exit_point in exit_points:
            idx = spread_zscore.index.get_loc(exit_point[0])
            ax3.scatter(exit_point[0], spread_zscore.iloc[idx], color='yellow', 
                       marker='o', s=80, alpha=0.8, edgecolors='black')
        
        ax3.set_title(f'Spread Z-Score (lookback={LOOKBACK_PERIOD}, min hold={MIN_HOLDING_PERIODS})')
        ax3.set_ylabel('Z-Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Output signal statistics
        print(f"\nSignal statistics for {pair_name}:")
        print(f"Long spread trades: {len(entry_points_long)}")
        print(f"Short spread trades: {len(entry_points_short)}")
        print(f"Closed trades: {len(exit_points)}")
        print(f"Minimum holding time: {MIN_HOLDING_PERIODS} periods ({MIN_HOLDING_PERIODS * 5} minutes)")
    
    def run_full_analysis(self):
        """
        Run full analysis
        """
        print("QUANTITATIVE PAIRS TRADING ANALYSIS")
        print("=" * 50)
        
        # Load data
        self.load_all_data()
        
        if self.prices_df is None or len(self.prices_df) == 0:
            print("Error: Failed to load data")
            return
        
        # 1. Correlation analysis
        top_20_pairs, stability_df = self.calculate_correlation_analysis()
        
        # 2. Cointegration tests
        eg_df = self.cointegration_tests(top_20_pairs)
        
        # 3. Filter pairs by criteria
        qualified_pairs = self.filter_qualified_pairs(top_20_pairs, stability_df, eg_df)
        
        # 4. Spread analysis
        if len(qualified_pairs) > 0:
            spread_results = self.spread_analysis(qualified_pairs)
        else:
            print("No pairs suitable for pairs trading by given criteria")


# Run analysis
if __name__ == "__main__":
    analyzer = PairsTradingAnalyzer()
    analyzer.run_full_analysis()
