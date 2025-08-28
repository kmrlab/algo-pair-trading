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
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Visualization settings
plt.style.use('dark_background')
sns.set_palette("viridis")

class HalfLifeAnalyzer:
    def __init__(self, data_dir='bybit_data', rolling_window=500):
        self.data_dir = Path(data_dir)
        self.rolling_window = rolling_window
        self.btc_data = None
        self.xrp_data = None
        self.spread_data = []
        self.half_life_results = {}
        self.output_dir = Path('half_life')
        self._ensure_output_directory()
        
    def _ensure_output_directory(self):
        """Create output directory for results"""
        try:
            self.output_dir.mkdir(exist_ok=True)
            print(f"📁 Results directory: {self.output_dir.resolve()}")
        except Exception as e:
            print(f"❌ Error creating directory {self.output_dir}: {e}")
    
    def load_data(self):
        """Load BTC and XRP data"""
        print("🔄 Loading BTC and XRP data...")
        
        try:
            # Load BTC
            btc_file = self.data_dir / 'BTCUSDT_5m_2024-01-01_to_2025-08-10.csv'
            self.btc_data = pd.read_csv(btc_file)
            print(f"✅ BTC: {len(self.btc_data):,} records")
            
            # Load XRP
            xrp_file = self.data_dir / 'XRPUSDT_5m_2024-01-01_to_2025-08-10.csv'
            self.xrp_data = pd.read_csv(xrp_file)
            print(f"✅ XRP: {len(self.xrp_data):,} records")
            
            # Synchronize by length
            min_length = min(len(self.btc_data), len(self.xrp_data))
            self.btc_data = self.btc_data.tail(min_length).reset_index(drop=True)
            self.xrp_data = self.xrp_data.tail(min_length).reset_index(drop=True)
            
            print(f"✅ Synchronized series: {min_length:,} observations")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def calculate_spread(self, window_size=1000):
        """
        Calculate spread with rolling hedge ratios
        
        Spread(t) = ln(P_BTC(t)) - β(t) × ln(P_XRP(t))
        """
        print(f"\n🔄 Calculating spread with rolling window of {window_size} observations...")
        
        # Logarithmic prices
        log_btc = np.log(self.btc_data['close'].values)
        log_xrp = np.log(self.xrp_data['close'].values)
        
        n = len(log_btc)
        spreads = []
        hedge_ratios = []
        residual_std = []
        
        # Progress bar for spread calculation
        progress_bar = tqdm(range(window_size, n), 
                           desc="📊 Calculating spread", 
                           bar_format="{l_bar}{bar:30}{r_bar}", colour='blue')
        
        for i in progress_bar:
            # Take rolling window
            btc_window = log_btc[i-window_size:i]
            xrp_window = log_xrp[i-window_size:i]
            
            # OLS regression for hedge ratio
            X = np.column_stack([np.ones(window_size), xrp_window])
            beta = np.linalg.lstsq(X, btc_window, rcond=None)[0]
            hedge_ratio = beta[1]
            
            # Spread at current moment
            spread = log_btc[i] - hedge_ratio * log_xrp[i]
            
            # Standard deviation of residuals for normalization
            residuals = btc_window - (beta[0] + beta[1] * xrp_window)
            std_residuals = np.std(residuals)
            
            spreads.append(spread)
            hedge_ratios.append(hedge_ratio)
            residual_std.append(std_residuals)
            
            # Update progress
            if i % 5000 == 0:
                progress_bar.set_postfix_str(f"β={hedge_ratio:.3f}")
        
        progress_bar.close()
        
        # Create DataFrame with results
        self.spread_data = pd.DataFrame({
            'spread': spreads,
            'hedge_ratio': hedge_ratios,
            'residual_std': residual_std,
            'timestamp': self.btc_data['timestamp'].iloc[window_size:].values if 'timestamp' in self.btc_data.columns else range(len(spreads))
        })
        
        print(f"✅ Calculated {len(spreads):,} spread values")
        
        # Basic spread statistics
        print(f"   • Average hedge ratio: {np.mean(hedge_ratios):.4f}")
        print(f"   • Std of hedge ratio: {np.std(hedge_ratios):.4f}")
        print(f"   • Average spread value: {np.mean(spreads):.6f}")
        print(f"   • Standard deviation of spread: {np.std(spreads):.6f}")
        
    def calculate_half_life_rolling(self):
        """
        Calculate half-life with given rolling window
        Using AR(1) model: Δspread(t) = α + λ×spread(t-1) + ε(t)
        Half-life = -ln(2) / ln(1 + λ)
        """
        print(f"\n🔄 Calculating spread half-life...")
        
        spreads = self.spread_data['spread'].values
        window = self.rolling_window
        
        print(f"\n🔢 Rolling window: {window} observations")
        
        half_lives = []
        lambda_coeffs = []
        valid_periods = []
        
        # Progress bar
        progress_bar = tqdm(range(window, len(spreads)), 
                           desc=f"📈 Half-life {window}p", 
                           bar_format="{l_bar}{bar:25}{r_bar}", colour='green')
        
        for i in progress_bar:
            spread_window = spreads[i-window:i]
            
            # Check stationarity
            try:
                adf_result = adfuller(spread_window, maxlag=int(12*(len(spread_window)/100)**(1/4)))
                if adf_result[1] > 0.10:  # If p-value > 0.10, spread is not stationary
                    continue
            except:
                continue
            
            # AR(1) model for half-life calculation
            try:
                # Δspread(t) = spread(t) - spread(t-1)
                delta_spread = np.diff(spread_window)
                lagged_spread = spread_window[:-1]
                
                # Regression: Δspread = α + λ×spread_lag + ε
                X = np.column_stack([np.ones(len(lagged_spread)), lagged_spread])
                coeffs = np.linalg.lstsq(X, delta_spread, rcond=None)[0]
                lambda_coeff = coeffs[1]
                
                # Half-life calculation
                if lambda_coeff < 0:  # Mean reversion condition
                    half_life = -np.log(2) / np.log(1 + lambda_coeff)
                    
                    # Convert to hours (5-minute intervals)
                    half_life_hours = half_life * 5 / 60
                    
                    if 0.1 <= half_life_hours <= 168:  # From 6 minutes to one week
                        half_lives.append(half_life_hours)
                        lambda_coeffs.append(lambda_coeff)
                        valid_periods.append(i)
                        
                        # Update progress
                        progress_bar.set_postfix_str(f"HL={half_life_hours:.1f}h")
                
            except Exception as e:
                continue
        
        progress_bar.close()
        
        # Statistics for window
        if half_lives:
            self.half_life_results[window] = {
                'half_lives': half_lives,
                'lambda_coeffs': lambda_coeffs,
                'valid_periods': valid_periods,
                'mean_half_life': np.mean(half_lives),
                'median_half_life': np.median(half_lives),
                'std_half_life': np.std(half_lives),
                'min_half_life': np.min(half_lives),
                'max_half_life': np.max(half_lives),
                'count_valid': len(half_lives)
            }
            
            print(f"   ✅ Valid calculations: {len(half_lives):,}")
            print(f"   • Average half-life: {np.mean(half_lives):.2f} hours")
            print(f"   • Median half-life: {np.median(half_lives):.2f} hours") 
            print(f"   • Standard deviation: {np.std(half_lives):.2f} hours")
            print(f"   • Range: {np.min(half_lives):.2f} - {np.max(half_lives):.2f} hours")
        else:
            print(f"   ❌ Failed to calculate valid half-life values")
    
    def print_summary(self):
        """Output final summary"""
        print("\n" + "="*80)
        print("📊 BTC/XRP SPREAD HALF-LIFE ANALYSIS SUMMARY")
        print("="*80)
        
        if not self.half_life_results:
            print("❌ No results to display")
            return
            
        # We only have one window, so simplify
        window = self.rolling_window
        if window not in self.half_life_results:
            print(f"❌ No results for window {window}")
            return
            
        results = self.half_life_results[window]
        
        print(f"🔢 Rolling window: {window} observations")
        print(f"   • Valid calculations: {results['count_valid']:,}")
        print(f"   • Mean: {results['mean_half_life']:.2f} hours")
        print(f"   • Median: {results['median_half_life']:.2f} hours")
        print(f"   • STD: {results['std_half_life']:.2f} hours")
        print(f"   • Range: {results['min_half_life']:.2f} - {results['max_half_life']:.2f} hours")
        
        print("\n📈 STRATEGY RECOMMENDATIONS:")
        print("-"*50)
        
        if results['count_valid'] > 100:
            print(f"✅ Median half-life: {results['median_half_life']:.2f} hours")
            print(f"   • Spread returns to mean in ~{results['median_half_life']:.1f} hours")
            
            # Trading parameter recommendations
            lookback_periods = int(results['median_half_life'] * 12 * 2)  # 2x median half-life in 5-min periods
            print(f"   • Recommended LOOKBACK_PERIOD: {lookback_periods} (≈{lookback_periods*5/60:.1f} hours)")
            print(f"   • Maximum time in position: {results['median_half_life']*2:.1f} hours")
            
            if results['median_half_life'] < 12:
                print("✅ Half-life < 12 hours - excellent for intraday trading")
            elif results['median_half_life'] < 24:
                print("⚠️  Half-life 12-24 hours - suitable but requires patience")
            else:
                print("❌ Half-life > 24 hours - too slow for active trading")
        else:
            print("❌ Insufficient valid calculations for reliable recommendations")
        
        print("="*80)
    
    def save_text_report(self):
        """Create and save text report"""
        if not self.half_life_results:
            print("❌ No data to create report")
            return None
            
        window = self.rolling_window
        if window not in self.half_life_results:
            print(f"❌ No results for window {window}")
            return None
            
        results = self.half_life_results[window]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'half_life_report_{window}w_{timestamp}.txt'
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("📊 BTC/XRP SPREAD HALF-LIFE ANALYSIS\n")
                f.write("="*80 + "\n")
                f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Rolling window: {window} observations\n\n")
                
                # Main statistics
                f.write("📈 HALF-LIFE STATISTICS:\n")
                f.write("-"*50 + "\n")
                f.write(f"• Valid calculations: {results['count_valid']:,}\n")
                f.write(f"• Mean: {results['mean_half_life']:.4f} hours\n")
                f.write(f"• Median: {results['median_half_life']:.4f} hours\n")
                f.write(f"• Standard deviation: {results['std_half_life']:.4f} hours\n")
                f.write(f"• Minimum: {results['min_half_life']:.4f} hours\n")
                f.write(f"• Maximum: {results['max_half_life']:.4f} hours\n\n")
                
                # Spread statistics (if available)
                if len(self.spread_data) > 0:
                    f.write("📊 SPREAD STATISTICS:\n")
                    f.write("-"*50 + "\n")
                    f.write(f"• Number of observations: {len(self.spread_data):,}\n")
                    f.write(f"• Mean value: {np.mean(self.spread_data['spread']):.6f}\n")
                    f.write(f"• Standard deviation: {np.std(self.spread_data['spread']):.6f}\n")
                    f.write(f"• Average hedge ratio: {np.mean(self.spread_data['hedge_ratio']):.6f}\n")
                    f.write(f"• Std of hedge ratio: {np.std(self.spread_data['hedge_ratio']):.6f}\n\n")
                
                # Trading recommendations
                f.write("📈 STRATEGY RECOMMENDATIONS:\n")
                f.write("-"*50 + "\n")
                
                if results['count_valid'] > 100:
                    f.write(f"✅ Median half-life: {results['median_half_life']:.2f} hours\n")
                    f.write(f"   • Spread returns to mean in ~{results['median_half_life']:.1f} hours\n")
                    
                    # Trading parameter recommendations
                    lookback_periods = int(results['median_half_life'] * 12 * 2)
                    f.write(f"   • Recommended LOOKBACK_PERIOD: {lookback_periods} (≈{lookback_periods*5/60:.1f} hours)\n")
                    f.write(f"   • Maximum time in position: {results['median_half_life']*2:.1f} hours\n\n")
                    
                    if results['median_half_life'] < 12:
                        f.write("✅ Half-life < 12 hours - excellent for intraday trading\n")
                    elif results['median_half_life'] < 24:
                        f.write("⚠️  Half-life 12-24 hours - suitable but requires patience\n")
                    else:
                        f.write("❌ Half-life > 24 hours - too slow for active trading\n")
                else:
                    f.write("❌ Insufficient valid calculations for reliable recommendations\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("📊 CALCULATION DETAILS:\n")
                f.write("="*80 + "\n")
                f.write(f"Method: AR(1) model for half-life calculation\n")
                f.write(f"Formula: Half-life = -ln(2) / ln(1 + λ)\n")
                f.write(f"Model: Δspread(t) = α + λ×spread(t-1) + ε(t)\n")
                f.write(f"Stationarity criterion: ADF test (p-value < 0.10)\n")
                f.write(f"Range filter: 0.1 ≤ half_life ≤ 168 hours\n")
                f.write("="*80 + "\n")
                
            print(f"✅ Text report saved: {filepath}")
            return filename
            
        except Exception as e:
            print(f"❌ Error saving report: {e}")
            return None
    
    def create_visualization(self):
        """Create half-life analysis plots"""
        if not self.half_life_results:
            print("❌ No data for visualization")
            return
            
        window = self.rolling_window
        if window not in self.half_life_results:
            print(f"❌ No results for window {window}")
            return
            
        print("\n🔄 Creating visualization...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'BTC/XRP Spread Half-Life Analysis (window {window})', fontsize=16, fontweight='bold')
        
        results = self.half_life_results[window]
        
        # 1. Spread over time plot
        if len(self.spread_data) > 0:
            sample_indices = np.linspace(0, len(self.spread_data)-1, min(5000, len(self.spread_data)), dtype=int)
            axes[0,0].plot(sample_indices, self.spread_data['spread'].iloc[sample_indices], 
                          alpha=0.7, color='cyan', linewidth=0.5)
            axes[0,0].axhline(y=self.spread_data['spread'].mean(), color='red', 
                             linestyle='--', alpha=0.8, label='Mean')
            axes[0,0].set_title('Spread Time Series')
            axes[0,0].set_xlabel('Period (5-min)')
            axes[0,0].set_ylabel('Spread')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
        
        # 2. Half-life distribution
        if results['half_lives']:
            axes[0,1].hist(results['half_lives'], bins=30, alpha=0.7, 
                          color='cyan', edgecolor='white', linewidth=0.5)
            axes[0,1].axvline(results['median_half_life'], color='yellow', 
                             linestyle='--', linewidth=2, label=f'Median: {results["median_half_life"]:.2f}h')
            axes[0,1].axvline(results['mean_half_life'], color='orange', 
                             linestyle='--', linewidth=2, label=f'Mean: {results["mean_half_life"]:.2f}h')
            axes[0,1].set_title('Half-Life Distribution')
            axes[0,1].set_xlabel('Half-life (hours)')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Half-life statistics
        stats_labels = ['Mean', 'Median', 'Min', 'Max']
        stats_values = [results['mean_half_life'], results['median_half_life'],
                       results['min_half_life'], results['max_half_life']]
        colors = ['cyan', 'yellow', 'green', 'red']
        
        bars = axes[1,0].bar(stats_labels, stats_values, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
        axes[1,0].set_title('Half-Life Statistics')
        axes[1,0].set_ylabel('Half-life (hours)')
        axes[1,0].grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, value in zip(bars, stats_values):
            height = bar.get_height()
            axes[1,0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                          f'{value:.2f}h', ha='center', va='bottom', fontweight='bold')
        
        # 4. Half-life dynamics over time
        if len(results['half_lives']) > 10:
            # Smoothing for clarity
            window_smooth = min(50, len(results['half_lives'])//10)
            smoothed = pd.Series(results['half_lives']).rolling(window=window_smooth, center=True).mean()
            
            axes[1,1].plot(smoothed, color='cyan', linewidth=2, alpha=0.8)
            axes[1,1].fill_between(range(len(smoothed)), smoothed, alpha=0.3, color='cyan')
            axes[1,1].axhline(results['median_half_life'], color='yellow', linestyle='--', alpha=0.8)
            axes[1,1].set_title(f'Half-Life Dynamics Over Time')
            axes[1,1].set_xlabel('Analysis Period')
            axes[1,1].set_ylabel('Half-life (hours)')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to half_life folder
        filename = f'half_life_analysis_{window}w_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        print(f"✅ Plot saved: {filepath}")
        
        plt.close()
        return filename  # Return filename for use in report
    
    def run_analysis(self):
        """Run full half-life analysis"""
        print("🚀 BTC/XRP SPREAD HALF-LIFE ANALYSIS")
        print("="*50)
        
        try:
            # 1. Load data
            if not self.load_data():
                return False
            
            # 2. Calculate spread with rolling hedge ratios
            self.calculate_spread(window_size=1000)
            
            # 3. Analyze half-life with given rolling window
            self.calculate_half_life_rolling()
            
            # 4. Output summary
            self.print_summary()
            
            # 5. Save reports
            print("\n🔄 Saving results...")
            
            # 5.1. Create and save visualization
            image_filename = self.create_visualization()
            
            # 5.2. Create and save text report
            report_filename = self.save_text_report()
            
            # 6. Final message
            print("\n🎉 Half-life analysis completed successfully!")
            if image_filename and report_filename:
                print(f"📁 Results saved in folder: {self.output_dir.resolve()}")
                print(f"   • Plot: {image_filename}")
                print(f"   • Report: {report_filename}")
            return True
            
        except Exception as e:
            print(f"❌ Analysis execution error: {e}")
            return False

def main(rolling_window=500):
    """
    Main function
    
    Args:
        rolling_window (int): Rolling window size for half-life analysis
    """
    print(f"🎯 Starting analysis with rolling window: {rolling_window} observations")
    analyzer = HalfLifeAnalyzer(rolling_window=rolling_window)
    analyzer.run_analysis()

if __name__ == "__main__":
    import sys
    
    # Ability to pass window size via command line argument
    rolling_window = 500  # Default
    
    if len(sys.argv) > 1:
        try:
            rolling_window = int(sys.argv[1])
            print(f"📝 Using rolling window from argument: {rolling_window}")
        except ValueError:
            print(f"⚠️  Invalid argument format '{sys.argv[1]}', using default value: {rolling_window}")
    
    # Reasonable value limits
    if rolling_window < 100:
        print(f"⚠️  Window {rolling_window} too small, set to minimum: 100")
        rolling_window = 100
    elif rolling_window > 10000:
        print(f"⚠️  Window {rolling_window} too large, set to maximum: 10000")
        rolling_window = 10000
    
    main(rolling_window)
