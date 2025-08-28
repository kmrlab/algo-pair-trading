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
from binance.client import Client
from datetime import datetime, timedelta
import time
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STRATEGY SETTINGS
# =============================================================================

# Time parameters
START_DATE = '2024-01-01'          # Analysis start date
END_DATE = '2024-01-15'            # Analysis end date
TIMEFRAME = '5m'                   # Data timeframe

# Trading pairs
ASSET1 = 'NEARUSDT'               # First asset
ASSET2 = 'DOTUSDT'                # Second asset

# Capital and risk management
INITIAL_CAPITAL = 10000           # Initial capital in USDT
LEVERAGE = 1                      # Leverage
TRADING_CAPITAL = INITIAL_CAPITAL * LEVERAGE  # Trading capital
COMMISSION_RATE = 0.0002          # Transaction commission (0.02%)

# Strategy parameters
LOOKBACK_PERIOD = 96              # Period for calculating rolling beta and Z-Score (96 = 8 hours on 5m)
ENTRY_THRESHOLD = 2.0             # Z-Score for position entry
EXIT_THRESHOLD = 0.5              # Z-Score for position exit
MIN_HOLDING_PERIODS = 4           # Minimum position holding time (4 periods = 20 minutes)

# API parameters
API_DELAY = 0.2                   # Delay between API requests (seconds)
KLINES_LIMIT = 1000               # Maximum candles per request

# =============================================================================

class PairsTradingStrategy:
    def __init__(self):
        """
        Initialize pairs trading strategy
        """
        # Binance client (public API, keys not required for data retrieval)
        self.client = Client()
        
        # Data
        self.asset1_data = None
        self.asset2_data = None
        self.prices_df = None
        
        # Backtesting results
        self.trades = []
        self.portfolio_value = []
        self.positions = []
        
    def fetch_futures_data(self, symbol, start_date=START_DATE, end_date=END_DATE, interval=TIMEFRAME):
        """
        Fetch futures data from Binance
        """
        try:
            print(f"Fetching data for {symbol}...")
            
            # Convert dates to milliseconds
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
            
            all_klines = []
            current_start = start_ts
            
            while current_start < end_ts:
                # Fetch data
                klines = self.client.futures_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=current_start,
                    endTime=end_ts,
                    limit=KLINES_LIMIT
                )
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                
                # Update start point
                last_timestamp = int(klines[-1][0])
                interval_ms = 5 * 60 * 1000  # 5 minutes in milliseconds
                current_start = last_timestamp + interval_ms
                
                # Pause between requests
                time.sleep(API_DELAY)
                
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
            
            # Remove duplicates and sort
            df = df[~df.index.duplicated(keep='first')]
            df = df.sort_index()
            
            print(f"  Loaded {len(df)} candles for {symbol}")
            print(f"  Period: {df.index[0]} - {df.index[-1]}")
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def load_data(self):
        """
        Load data for both assets
        """
        print("Loading data for pairs trading...")
        print(f"Pair: {ASSET1} - {ASSET2}")
        print(f"Period: {START_DATE} - {END_DATE}")
        print(f"Timeframe: {TIMEFRAME}")
        
        # Load data
        self.asset1_data = self.fetch_futures_data(ASSET1)
        self.asset2_data = self.fetch_futures_data(ASSET2)
        
        if self.asset1_data is None or self.asset2_data is None:
            raise ValueError("Failed to load data for one or both assets")
        
        # Create common DataFrame
        self.prices_df = pd.DataFrame({
            ASSET1: self.asset1_data['close'],
            ASSET2: self.asset2_data['close']
        })
        
        # Remove missing values
        self.prices_df.dropna(inplace=True)
        
        print(f"\nData loaded successfully")
        print(f"Number of observations: {len(self.prices_df)}")
        print(f"Data period: {self.prices_df.index[0]} - {self.prices_df.index[-1]}")
    
    def calculate_rolling_beta(self, asset1_prices, asset2_prices, lookback_period):
        """
        Calculate rolling beta using linear regression
        """
        betas = []
        
        for i in range(lookback_period, len(asset1_prices)):
            # Take data window
            y = asset1_prices.iloc[i-lookback_period:i].values
            X = asset2_prices.iloc[i-lookback_period:i].values.reshape(-1, 1)
            
            # Linear regression
            reg = LinearRegression().fit(X, y)
            betas.append(reg.coef_[0])
        
        # Create Series with correct index
        beta_series = pd.Series(betas, index=asset1_prices.index[lookback_period:])
        
        return beta_series
    
    def calculate_signals(self):
        """
        Calculate trading signals based on spread Z-Score
        """
        print("\nCalculating trading signals...")
        
        # Get prices
        asset1_prices = self.prices_df[ASSET1]
        asset2_prices = self.prices_df[ASSET2]
        
        # Calculate rolling beta
        rolling_beta = self.calculate_rolling_beta(asset1_prices, asset2_prices, LOOKBACK_PERIOD)
        
        # Calculate spread with rolling beta
        common_index = rolling_beta.index
        spread = asset1_prices.loc[common_index] - rolling_beta * asset2_prices.loc[common_index]
        
        # Calculate Z-Score
        spread_mean = spread.rolling(LOOKBACK_PERIOD).mean()
        spread_std = spread.rolling(LOOKBACK_PERIOD).std()
        spread_zscore = (spread - spread_mean) / spread_std
        
        # Remove NaN values
        spread_zscore = spread_zscore.dropna()
        
        # Save results
        self.spread = spread.loc[spread_zscore.index]
        self.spread_zscore = spread_zscore
        self.rolling_beta = rolling_beta.loc[spread_zscore.index]
        
        print(f"Signals calculated for {len(self.spread_zscore)} periods")
        print(f"Average beta: {self.rolling_beta.mean():.6f}")
        print(f"Beta std dev: {self.rolling_beta.std():.6f}")
    
    def backtest(self):
        """
        Strategy backtesting
        """
        print("\nStarting backtesting...")
        
        # Initialization
        capital = INITIAL_CAPITAL
        position = 0  # 0 - no position, 1 - long spread, -1 - short spread
        periods_in_position = 0
        entry_price1 = 0
        entry_price2 = 0
        entry_beta = 0
        
        # History
        equity_curve = []
        drawdown_curve = []
        max_equity = INITIAL_CAPITAL
        
        # Get common index
        common_index = self.spread_zscore.index
        asset1_prices = self.prices_df[ASSET1].loc[common_index]
        asset2_prices = self.prices_df[ASSET2].loc[common_index]
        
        for i in range(len(self.spread_zscore)):
            current_time = self.spread_zscore.index[i]
            current_zscore = self.spread_zscore.iloc[i]
            current_beta = self.rolling_beta.iloc[i]
            current_price1 = asset1_prices.iloc[i]
            current_price2 = asset2_prices.iloc[i]
            
            # Entry and exit logic
            if position == 0:  # No position
                if current_zscore > ENTRY_THRESHOLD:  # Spread high - short spread
                    # Short ASSET1, Long ASSET2
                    position = -1
                    periods_in_position = 0
                    entry_price1 = current_price1
                    entry_price2 = current_price2
                    entry_beta = current_beta
                    
                    # Calculate positions considering beta
                    # Distribute capital proportionally to beta
                    total_value = TRADING_CAPITAL
                    asset2_value = total_value / (1 + current_beta)
                    asset1_value = total_value - asset2_value
                    
                    position_size1 = asset1_value / current_price1
                    position_size2 = asset2_value / current_price2
                    
                    # Entry commission
                    commission = TRADING_CAPITAL * COMMISSION_RATE * 2  # For both assets
                    capital -= commission
                    
                    self.trades.append({
                        'entry_time': current_time,
                        'type': 'SHORT_SPREAD',
                        'entry_price1': entry_price1,
                        'entry_price2': entry_price2,
                        'entry_beta': entry_beta,
                        'entry_zscore': current_zscore,
                        'position_size1': position_size1,
                        'position_size2': position_size2,
                        'direction1': 'SHORT',
                        'direction2': 'LONG'
                    })
                    
                elif current_zscore < -ENTRY_THRESHOLD:  # Spread low - long spread
                    # Long ASSET1, Short ASSET2
                    position = 1
                    periods_in_position = 0
                    entry_price1 = current_price1
                    entry_price2 = current_price2
                    entry_beta = current_beta
                    
                    # Calculate positions
                    total_value = TRADING_CAPITAL
                    asset2_value = total_value / (1 + current_beta)
                    asset1_value = total_value - asset2_value
                    
                    position_size1 = asset1_value / current_price1
                    position_size2 = asset2_value / current_price2
                    
                    # Entry commission
                    commission = TRADING_CAPITAL * COMMISSION_RATE * 2
                    capital -= commission
                    
                    self.trades.append({
                        'entry_time': current_time,
                        'type': 'LONG_SPREAD',
                        'entry_price1': entry_price1,
                        'entry_price2': entry_price2,
                        'entry_beta': entry_beta,
                        'entry_zscore': current_zscore,
                        'position_size1': position_size1,
                        'position_size2': position_size2,
                        'direction1': 'LONG',
                        'direction2': 'SHORT'
                    })
            
            elif position != 0:  # Has position
                periods_in_position += 1
                
                # Calculate current profit/loss
                if position == 1:  # Long spread
                    # Profit from long ASSET1
                    pnl1 = self.trades[-1]['position_size1'] * (current_price1 - entry_price1)
                    # Profit from short ASSET2
                    pnl2 = self.trades[-1]['position_size2'] * (entry_price2 - current_price2)
                else:  # Short spread
                    # Profit from short ASSET1
                    pnl1 = self.trades[-1]['position_size1'] * (entry_price1 - current_price1)
                    # Profit from long ASSET2
                    pnl2 = self.trades[-1]['position_size2'] * (current_price2 - entry_price2)
                
                total_pnl = pnl1 + pnl2
                
                # Exit position
                if periods_in_position >= MIN_HOLDING_PERIODS and abs(current_zscore) < EXIT_THRESHOLD:
                    # Exit commission
                    commission = TRADING_CAPITAL * COMMISSION_RATE * 2
                    capital = capital + total_pnl - commission
                    
                    # Save trade information
                    self.trades[-1].update({
                        'exit_time': current_time,
                        'exit_price1': current_price1,
                        'exit_price2': current_price2,
                        'exit_zscore': current_zscore,
                        'pnl': total_pnl,
                        'pnl_percent': (total_pnl / TRADING_CAPITAL) * 100,
                        'holding_periods': periods_in_position,
                        'commission_total': TRADING_CAPITAL * COMMISSION_RATE * 4  # Entry + exit
                    })
                    
                    position = 0
                    periods_in_position = 0
            
            # Update equity
            current_equity = capital
            if position != 0:
                # Add unrealized profit
                if position == 1:
                    unrealized_pnl1 = self.trades[-1]['position_size1'] * (current_price1 - entry_price1)
                    unrealized_pnl2 = self.trades[-1]['position_size2'] * (entry_price2 - current_price2)
                else:
                    unrealized_pnl1 = self.trades[-1]['position_size1'] * (entry_price1 - current_price1)
                    unrealized_pnl2 = self.trades[-1]['position_size2'] * (current_price2 - entry_price2)
                current_equity += unrealized_pnl1 + unrealized_pnl2
            
            equity_curve.append(current_equity)
            
            # Calculate drawdown
            if current_equity > max_equity:
                max_equity = current_equity
            drawdown = (current_equity - max_equity) / max_equity * 100
            drawdown_curve.append(drawdown)
            
            # Save positions for visualization
            self.positions.append({
                'time': current_time,
                'position': position,
                'equity': current_equity,
                'drawdown': drawdown
            })
        
        # Save results
        self.equity_curve = pd.Series(equity_curve, index=common_index)
        self.drawdown_curve = pd.Series(drawdown_curve, index=common_index)
        
        print(f"\nBacktesting completed")
        print(f"Number of trades: {len([t for t in self.trades if 'exit_time' in t])}")
    
    def calculate_metrics(self):
        """
        Calculate performance metrics
        """
        print("\n=== PERFORMANCE METRICS ===")
        
        # Filter only closed trades
        closed_trades = [t for t in self.trades if 'exit_time' in t]
        
        if len(closed_trades) == 0:
            print("No closed trades for analysis")
            return
        
        # Basic metrics
        total_return = (self.equity_curve.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        max_drawdown = self.drawdown_curve.min()
        
        # Trade metrics
        profits = [t['pnl'] for t in closed_trades]
        profit_percents = [t['pnl_percent'] for t in closed_trades]
        
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        
        win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0
        
        avg_profit = np.mean(profits) if profits else 0
        avg_profit_percent = np.mean(profit_percents) if profit_percents else 0
        
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        # Profit Factor
        total_wins = sum(winning_trades) if winning_trades else 0
        total_losses = abs(sum(losing_trades)) if losing_trades else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Sharpe Ratio (annualized)
        returns = self.equity_curve.pct_change().dropna()
        periods_per_year = 365 * 24 * 12  # 5-minute periods per year
        sharpe_ratio = np.sqrt(periods_per_year) * returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Average position holding time
        holding_times = [t['holding_periods'] for t in closed_trades]
        avg_holding_time = np.mean(holding_times) if holding_times else 0
        
        # Output metrics
        print(f"\nGENERAL METRICS:")
        print(f"Initial capital: {INITIAL_CAPITAL:,.2f} USDT")
        print(f"Final capital: {self.equity_curve.iloc[-1]:,.2f} USDT")
        print(f"Total return: {total_return:.2f}%")
        print(f"Maximum drawdown: {max_drawdown:.2f}%")
        print(f"Sharpe ratio: {sharpe_ratio:.2f}")
        
        print(f"\nTRADE METRICS:")
        print(f"Total trades: {len(closed_trades)}")
        print(f"Profitable trades: {len(winning_trades)}")
        print(f"Losing trades: {len(losing_trades)}")
        print(f"Win rate: {win_rate:.2f}%")
        print(f"Profit factor: {profit_factor:.2f}")
        
        print(f"\nAVERAGE METRICS:")
        print(f"Average profit per trade: {avg_profit:.2f} USDT ({avg_profit_percent:.2f}%)")
        print(f"Average winning trade: {avg_win:.2f} USDT")
        print(f"Average losing trade: {avg_loss:.2f} USDT")
        print(f"Average holding time: {avg_holding_time:.1f} periods ({avg_holding_time * 5:.0f} minutes)")
        
        print(f"\nTRADING PARAMETERS:")
        print(f"Leverage: {LEVERAGE}x")
        print(f"Trading capital: {TRADING_CAPITAL:,.2f} USDT")
        print(f"Commission: {COMMISSION_RATE * 100:.2f}%")
        
        # Details by trade types
        long_spread_trades = [t for t in closed_trades if t['type'] == 'LONG_SPREAD']
        short_spread_trades = [t for t in closed_trades if t['type'] == 'SHORT_SPREAD']
        
        print(f"\nTRADE TYPE DETAILS:")
        print(f"Long spread trades: {len(long_spread_trades)}")
        print(f"Short spread trades: {len(short_spread_trades)}")
    
    def plot_results(self):
        """
        Visualize backtesting results
        """
        print("\nCreating visualizations...")
        
        # Prepare data
        common_index = self.spread_zscore.index
        asset1_prices = self.prices_df[ASSET1].loc[common_index]
        asset2_prices = self.prices_df[ASSET2].loc[common_index]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 20))
        
        # 1. Asset prices with signals
        ax1 = plt.subplot(5, 1, 1)
        
        # Normalize prices for visualization
        price1_norm = (asset1_prices / asset1_prices.iloc[0]) * 100
        price2_norm = (asset2_prices / asset2_prices.iloc[0]) * 100
        
        ax1.plot(price1_norm.index, price1_norm, label=ASSET1, linewidth=1.5, alpha=0.8)
        ax1.plot(price2_norm.index, price2_norm, label=ASSET2, linewidth=1.5, alpha=0.8)
        
        # Display entry and exit signals
        for trade in self.trades:
            if 'exit_time' not in trade:
                continue
                
            # Trade entry
            entry_idx = price1_norm.index.get_loc(trade['entry_time'])
            if trade['type'] == 'LONG_SPREAD':
                ax1.scatter(trade['entry_time'], price1_norm.iloc[entry_idx], 
                          color='green', marker='^', s=100, alpha=0.8, zorder=5)
                ax1.scatter(trade['entry_time'], price2_norm.iloc[entry_idx], 
                          color='red', marker='v', s=100, alpha=0.8, zorder=5)
            else:  # SHORT_SPREAD
                ax1.scatter(trade['entry_time'], price1_norm.iloc[entry_idx], 
                          color='red', marker='v', s=100, alpha=0.8, zorder=5)
                ax1.scatter(trade['entry_time'], price2_norm.iloc[entry_idx], 
                          color='green', marker='^', s=100, alpha=0.8, zorder=5)
            
            # Trade exit
            exit_idx = price1_norm.index.get_loc(trade['exit_time'])
            ax1.scatter(trade['exit_time'], price1_norm.iloc[exit_idx], 
                      color='yellow', marker='o', s=80, alpha=0.8, edgecolors='black', zorder=5)
            ax1.scatter(trade['exit_time'], price2_norm.iloc[exit_idx], 
                      color='yellow', marker='o', s=80, alpha=0.8, edgecolors='black', zorder=5)
        
        ax1.set_title(f'Asset Prices with Trading Signals: {ASSET1} - {ASSET2}', fontsize=14)
        ax1.set_ylabel('Normalized Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add legend for signals
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], marker='^', color='w', markerfacecolor='g', markersize=10),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='r', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='y', 
                  markeredgecolor='black', markersize=10)
        ]
        ax1.legend(custom_lines + ax1.get_lines()[:2], 
                  ['Long', 'Short', 'Exit', ASSET1, ASSET2], 
                  loc='upper left')
        
        # 2. Rolling beta
        ax2 = plt.subplot(5, 1, 2)
        ax2.plot(self.rolling_beta.index, self.rolling_beta, linewidth=1.5, color='purple')
        ax2.axhline(y=self.rolling_beta.mean(), color='purple', linestyle='--', alpha=0.5, 
                   label=f'Mean: {self.rolling_beta.mean():.4f}')
        ax2.set_title(f'Rolling Beta (lookback={LOOKBACK_PERIOD})', fontsize=14)
        ax2.set_ylabel('Beta')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Spread Z-Score
        ax3 = plt.subplot(5, 1, 3)
        ax3.plot(self.spread_zscore.index, self.spread_zscore, linewidth=1, alpha=0.7, color='blue')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.axhline(y=ENTRY_THRESHOLD, color='red', linestyle='--', alpha=0.7, 
                   label=f'Entry ±{ENTRY_THRESHOLD}σ')
        ax3.axhline(y=-ENTRY_THRESHOLD, color='red', linestyle='--', alpha=0.7)
        ax3.axhline(y=EXIT_THRESHOLD, color='orange', linestyle=':', alpha=0.7, 
                   label=f'Exit ±{EXIT_THRESHOLD}σ')
        ax3.axhline(y=-EXIT_THRESHOLD, color='orange', linestyle=':', alpha=0.7)
        
        # Display signals on Z-Score
        for trade in self.trades:
            if 'exit_time' not in trade:
                continue
            
            # Entry
            entry_idx = self.spread_zscore.index.get_loc(trade['entry_time'])
            ax3.scatter(trade['entry_time'], trade['entry_zscore'], 
                      color='green' if trade['type'] == 'LONG_SPREAD' else 'red', 
                      marker='^' if trade['type'] == 'LONG_SPREAD' else 'v', 
                      s=100, alpha=0.8, zorder=5)
            
            # Exit
            exit_idx = self.spread_zscore.index.get_loc(trade['exit_time'])
            ax3.scatter(trade['exit_time'], trade['exit_zscore'], 
                      color='yellow', marker='o', s=80, alpha=0.8, 
                      edgecolors='black', zorder=5)
        
        ax3.set_title(f'Spread Z-Score', fontsize=14)
        ax3.set_ylabel('Z-Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Equity curve
        ax4 = plt.subplot(5, 1, 4)
        ax4.plot(self.equity_curve.index, self.equity_curve, linewidth=2, color='green')
        ax4.axhline(y=INITIAL_CAPITAL, color='black', linestyle='--', alpha=0.5, 
                   label=f'Initial Capital: {INITIAL_CAPITAL}')
        ax4.fill_between(self.equity_curve.index, INITIAL_CAPITAL, self.equity_curve, 
                        where=(self.equity_curve >= INITIAL_CAPITAL), 
                        color='green', alpha=0.3, label='Profit')
        ax4.fill_between(self.equity_curve.index, INITIAL_CAPITAL, self.equity_curve, 
                        where=(self.equity_curve < INITIAL_CAPITAL), 
                        color='red', alpha=0.3, label='Loss')
        
        ax4.set_title('Equity Curve', fontsize=14)
        ax4.set_ylabel('Capital (USDT)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Drawdown chart
        ax5 = plt.subplot(5, 1, 5)
        ax5.fill_between(self.drawdown_curve.index, 0, self.drawdown_curve, 
                        color='red', alpha=0.5)
        ax5.plot(self.drawdown_curve.index, self.drawdown_curve, 
                linewidth=1.5, color='darkred')
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Mark maximum drawdown
        max_dd = self.drawdown_curve.min()
        max_dd_idx = self.drawdown_curve.idxmin()
        ax5.scatter(max_dd_idx, max_dd, color='red', s=100, zorder=5)
        ax5.annotate(f'Max DD: {max_dd:.2f}%', 
                    xy=(max_dd_idx, max_dd), 
                    xytext=(10, 10), 
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax5.set_title('Drawdown Chart', fontsize=14)
        ax5.set_ylabel('Drawdown (%)')
        ax5.set_xlabel('Time')
        ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Additional visualization: profit/loss distribution
        if self.trades and any('pnl' in t for t in self.trades):
            fig2, (ax6, ax7) = plt.subplots(1, 2, figsize=(14, 6))
            
            # P&L histogram
            closed_trades = [t for t in self.trades if 'exit_time' in t]
            pnls = [t['pnl'] for t in closed_trades]
            pnl_percents = [t['pnl_percent'] for t in closed_trades]
            
            ax6.hist(pnls, bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax6.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax6.axvline(x=np.mean(pnls), color='green', linestyle='-', 
                       label=f'Mean: {np.mean(pnls):.2f}')
            ax6.set_title('Profit/Loss Distribution (USDT)')
            ax6.set_xlabel('P&L (USDT)')
            ax6.set_ylabel('Number of Trades')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            
            # P&L percentage histogram
            ax7.hist(pnl_percents, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax7.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax7.axvline(x=np.mean(pnl_percents), color='blue', linestyle='-', 
                       label=f'Mean: {np.mean(pnl_percents):.2f}%')
            ax7.set_title('Profit/Loss Distribution (%)')
            ax7.set_xlabel('P&L (%)')
            ax7.set_ylabel('Number of Trades')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def run_strategy(self):
        """
        Run complete strategy
        """
        print("=" * 60)
        print("RUNNING PAIRS ARBITRAGE STRATEGY")
        print("=" * 60)
        print(f"Pair: {ASSET1} - {ASSET2}")
        print(f"Period: {START_DATE} - {END_DATE}")
        print(f"Timeframe: {TIMEFRAME}")
        print(f"Initial capital: {INITIAL_CAPITAL} USDT")
        print(f"Leverage: {LEVERAGE}x")
        print("=" * 60)
        
        try:
            # 1. Load data
            self.load_data()
            
            # 2. Calculate signals
            self.calculate_signals()
            
            # 3. Backtesting
            self.backtest()
            
            # 4. Calculate metrics
            self.calculate_metrics()
            
            # 5. Visualization
            self.plot_results()
            
            print("\nStrategy executed successfully!")
            
        except Exception as e:
            print(f"\nError executing strategy: {e}")
            import traceback
            traceback.print_exc()


# Run strategy
if __name__ == "__main__":
    strategy = PairsTradingStrategy()
    strategy.run_strategy()
