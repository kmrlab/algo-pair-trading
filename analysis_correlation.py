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
import scipy.stats as stats
from pathlib import Path
from itertools import combinations
import warnings
import os
from datetime import datetime
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Visualization settings
plt.style.use('dark_background')
sns.set_palette("RdYlBu_r")

class CorrelationAnalyzer:
    def __init__(self, data_dir='bybit_data', output_dir='correlation'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.assets = {}
        self.results = {}
        self.correlation_matrix = None
        self.analysis_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def load_data(self):
        """Load all CSV files from directory"""
        print("🔄 Loading data...")
        
        csv_files = list(self.data_dir.glob('*.csv'))
        if not csv_files:
            raise FileNotFoundError(f"CSV files not found in {self.data_dir}")
            
        # Progress bar for loading files
        progress_bar = tqdm(csv_files, desc="📂 Loading files", 
                           bar_format="{l_bar}{bar:30}{r_bar}", colour='green')
        
        for file_path in progress_bar:
            # Extract asset name from filename
            asset_name = file_path.stem.split('_')[0]  # BTCUSDT -> BTC
            if asset_name.endswith('USDT'):
                asset_name = asset_name[:-4]  # BTCUSDT -> BTC
            
            # Update progress bar description
            progress_bar.set_postfix_str(f"Loading {asset_name}")
            
            try:
                df = pd.read_csv(file_path)
                
                # Check for required columns
                if 'close' not in df.columns:
                    tqdm.write(f"   ⚠️  Column 'close' not found in {file_path.name}")
                    continue
                    
                # Convert timestamp if available
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                elif 'time' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['time'])
                else:
                    # Create sequential index
                    df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='5T')
                
                # Calculate logarithmic returns with outlier handling
                prices = df['close'].values
                
                # Preliminary price cleaning from outliers (winsorization)
                prices_clean = stats.mstats.winsorize(prices, limits=(0.01, 0.01))
                log_returns = np.diff(np.log(prices_clean))  # r(t) = ln(P(t)/P(t-1))
                
                # Additional filtering of extreme returns
                log_returns = self._filter_extreme_returns(log_returns)
                
                self.assets[asset_name] = {
                    'data': df,
                    'prices': prices,
                    'log_returns': log_returns,
                    'timestamps': df['timestamp'].values if 'timestamp' in df.columns else np.arange(len(df))
                }
                
                tqdm.write(f"   ✅ {asset_name}: {len(log_returns):,} returns")
                
            except Exception as e:
                tqdm.write(f"   ❌ Error loading {file_path.name}: {e}")
                continue
        
        progress_bar.close()
        print(f"✅ Loaded assets: {', '.join(self.assets.keys())}")
        return len(self.assets) > 1
    
    def _filter_extreme_returns(self, returns, z_threshold=5.0):
        """
        Filter extreme returns based on Z-score
        """
        if len(returns) == 0:
            return returns
            
        # Calculate Z-score for each return
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return returns
        
        z_scores = np.abs((returns - mean_return) / std_return)
        
        # Replace extreme values with median
        median_return = np.median(returns)
        filtered_returns = np.where(z_scores > z_threshold, median_return, returns)
        
        return filtered_returns
        
    def sync_data(self):
        """Improved time series synchronization of returns"""
        print("\n🔄 Time series synchronization...")
        
        if len(self.assets) < 2:
            raise ValueError("Need at least 2 assets for correlation analysis")
            
        # Find minimum length among all assets (returns are 1 less than prices)
        min_length = min(len(asset['log_returns']) for asset in self.assets.values())
        
        # Statistics before synchronization
        original_lengths = {name: len(asset['log_returns']) for name, asset in self.assets.items()}
        print(f"   Original series lengths: {original_lengths}")
        
        # Trim all series to same length (taking latest data)
        filtered_count = {}
        for asset_name in self.assets:
            original_returns = self.assets[asset_name]['log_returns']
            synced_returns = original_returns[-min_length:]
            
            # Count filtered values
            if hasattr(original_returns, '__len__') and hasattr(synced_returns, '__len__'):
                filtered_count[asset_name] = len(original_returns) - len(synced_returns)
            else:
                filtered_count[asset_name] = 0
                
            self.assets[asset_name]['log_returns'] = synced_returns
        
        print(f"✅ Synchronized return series with length of {min_length} observations")
        if any(count > 0 for count in filtered_count.values()):
            print(f"   Filtered values: {filtered_count}")
        
    def calculate_correlations(self):
        """Calculate correlations between all asset pairs"""
        print("\n🔄 Calculating correlations of all asset pairs...")
        
        asset_names = list(self.assets.keys())
        n_assets = len(asset_names)
        
        # Create correlation matrix
        self.correlation_matrix = np.zeros((n_assets, n_assets))
        
        # Get all unique pairs for processing
        pairs_to_process = list(combinations(range(n_assets), 2))
        total_pairs = len(pairs_to_process)
        
        print(f"   Total pairs for analysis: {total_pairs}")
        
        # Progress bar for correlation calculation
        progress_bar = tqdm(pairs_to_process, desc="🔢 Calculating correlations", 
                           bar_format="{l_bar}{bar:30}{r_bar}{postfix}", colour='blue')
        
        # Fill matrix and results
        for i in range(n_assets):
            for j in range(n_assets):
                if i == j:
                    # Diagonal - self-correlation
                    self.correlation_matrix[i, j] = 1.0
        
        # Process only unique pairs
        for i, j in progress_bar:
            asset1 = asset_names[i]
            asset2 = asset_names[j]
            
            # Update progress bar description
            progress_bar.set_postfix_str(f"{asset1}-{asset2}")
            
            returns1 = self.assets[asset1]['log_returns']
            returns2 = self.assets[asset2]['log_returns']
            
            try:
                # Check for NaN and infinite values
                if np.any(np.isnan(returns1)) or np.any(np.isnan(returns2)) or \
                   np.any(np.isinf(returns1)) or np.any(np.isinf(returns2)):
                    tqdm.write(f"   ⚠️  Found NaN/Inf values in {asset1}-{asset2}")
                    correlation = 0.0
                else:
                    # Pearson correlation
                    correlation = np.corrcoef(returns1, returns2)[0, 1]
                    
                    # Check result for correctness
                    if np.isnan(correlation):
                        correlation = 0.0
                
                # Additional statistics with mini progress bars
                rolling_30d = self._calculate_rolling_correlation(returns1, returns2, window=30*24*12, desc=f"{asset1}-{asset2} 30d")
                rolling_7d = self._calculate_rolling_correlation(returns1, returns2, window=7*24*12, desc=f"{asset1}-{asset2} 7d")  
                rolling_1d = self._calculate_rolling_correlation(returns1, returns2, window=24*12, desc=f"{asset1}-{asset2} 1d")
                
                # Save to matrix (symmetrically)
                self.correlation_matrix[i, j] = correlation
                self.correlation_matrix[j, i] = correlation
                
                # Save detailed results
                self.results[f"{asset1}-{asset2}"] = {
                    'pair': f"{asset1}/{asset2}",
                    'correlation': correlation,
                    'correlation_abs': abs(correlation),
                    'rolling_30d_avg': np.mean(rolling_30d) if len(rolling_30d) > 0 else correlation,
                    'rolling_7d_avg': np.mean(rolling_7d) if len(rolling_7d) > 0 else correlation,
                    'rolling_1d_avg': np.mean(rolling_1d) if len(rolling_1d) > 0 else correlation,
                    'correlation_stability': self._assess_correlation_stability(rolling_30d),
                    'asset1': asset1,
                    'asset2': asset2
                }
                
            except Exception as e:
                tqdm.write(f"   ⚠️  Error calculating correlation {asset1}-{asset2}: {e}")
                self.correlation_matrix[i, j] = 0
                self.correlation_matrix[j, i] = 0
        
        progress_bar.close()
        print(f"✅ Correlation calculation completed for {len(self.results)} pairs")
        
    def _calculate_rolling_correlation(self, returns1, returns2, window, desc="Rolling correlation"):
        """Rolling correlation with progress bar"""
        if len(returns1) < window:
            return []
        
        # Number of iterations for rolling window
        n_iterations = len(returns1) - window
        
        # Show progress bar only for long operations (> 1000 iterations)
        if n_iterations > 1000:
            correlations = []
            iterations_range = tqdm(range(window, len(returns1)), 
                                  desc=f"📊 {desc}", leave=False,
                                  bar_format="{desc}: {percentage:3.0f}%|{bar:20}{r_bar}",
                                  colour='cyan', disable=False)
            
            for i in iterations_range:
                window_returns1 = returns1[i-window:i]
                window_returns2 = returns2[i-window:i]
                
                # Check for sufficient data variation
                if np.std(window_returns1) == 0 or np.std(window_returns2) == 0:
                    continue
                    
                # Check for NaN/Inf
                if np.any(np.isnan(window_returns1)) or np.any(np.isnan(window_returns2)) or \
                   np.any(np.isinf(window_returns1)) or np.any(np.isinf(window_returns2)):
                    continue
                
                corr = np.corrcoef(window_returns1, window_returns2)[0, 1]
                if not np.isnan(corr) and not np.isinf(corr):
                    correlations.append(corr)
        else:
            # For small data volumes don't show progress bar
            correlations = []
            for i in range(window, len(returns1)):
                window_returns1 = returns1[i-window:i]
                window_returns2 = returns2[i-window:i]
                
                # Check for sufficient data variation
                if np.std(window_returns1) == 0 or np.std(window_returns2) == 0:
                    continue
                    
                # Check for NaN/Inf
                if np.any(np.isnan(window_returns1)) or np.any(np.isnan(window_returns2)) or \
                   np.any(np.isinf(window_returns1)) or np.any(np.isinf(window_returns2)):
                    continue
                
                corr = np.corrcoef(window_returns1, window_returns2)[0, 1]
                if not np.isnan(corr) and not np.isinf(corr):
                    correlations.append(corr)
        
        return correlations
    
    def _assess_correlation_stability(self, rolling_correlations):
        """Assess correlation stability"""
        if len(rolling_correlations) < 10:
            return "insufficient_data"
        
        std_corr = np.std(rolling_correlations)
        
        if std_corr < 0.05:
            return "very_stable"
        elif std_corr < 0.10:
            return "stable"
        elif std_corr < 0.20:
            return "moderate"
        else:
            return "unstable"
    
    def create_output_directory(self):
        """Create output directory for results"""
        try:
            self.output_dir.mkdir(exist_ok=True)
            print(f"📁 Created/found results directory: {self.output_dir}")
        except Exception as e:
            print(f"❌ Error creating directory {self.output_dir}: {e}")
            
    def print_results(self):
        """Output results to terminal"""
        print("\n" + "="*90)
        print("📊 CORRELATION ANALYSIS RESULTS")
        print("="*90)
        
        # Sort by descending absolute correlation
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['correlation_abs'],
            reverse=True
        )
        
        print(f"{'№':<3} {'Pair':<12} {'Correlation':<12} {'|Corr|':<8} {'30d avg':<8} {'7d avg':<8} {'1d avg':<8} {'Stability':<15}")
        print("-" * 90)
        
        for i, (pair_name, result) in enumerate(sorted_results, 1):
            correlation_str = f"{result['correlation']:+.4f}"
            
            print(f"{i:<3} {result['pair']:<12} {correlation_str:<12} "
                  f"{result['correlation_abs']:<8.4f} {result['rolling_30d_avg']:<8.4f} "
                  f"{result['rolling_7d_avg']:<8.4f} {result['rolling_1d_avg']:<8.4f} "
                  f"{result['correlation_stability']:<15}")
                  
        # Statistics
        strong_positive = sum(1 for r in self.results.values() if r['correlation'] > 0.7)
        strong_negative = sum(1 for r in self.results.values() if r['correlation'] < -0.7)
        moderate_positive = sum(1 for r in self.results.values() if 0.4 <= r['correlation'] <= 0.7)
        moderate_negative = sum(1 for r in self.results.values() if -0.7 <= r['correlation'] <= -0.4)
        weak = sum(1 for r in self.results.values() if abs(r['correlation']) < 0.4)
        
        print("\n" + "="*90)
        print("📈 CORRELATION STATISTICS:")
        print(f"   • Total pairs analyzed: {len(self.results)}")
        print(f"   • Strong positive correlation (> 0.7): {strong_positive}")
        print(f"   • Strong negative correlation (< -0.7): {strong_negative}")
        print(f"   • Moderate positive (0.4 - 0.7): {moderate_positive}")
        print(f"   • Moderate negative (-0.7 - -0.4): {moderate_negative}")
        print(f"   • Weak correlation (|r| < 0.4): {weak}")
        
        if sorted_results:
            best_pair = sorted_results[0]
            print(f"   • Strongest correlation: {best_pair[1]['pair']} (r = {best_pair[1]['correlation']:+.4f})")
        
        print("="*90)
    
    def save_report(self):
        """Save text report"""
        print("\n🔄 Saving text report...")
        
        try:
            report_filename = self.output_dir / f'correlation_report_{self.analysis_timestamp}.txt'
            
            # Progress bar for saving report
            with tqdm(total=100, desc="💾 Creating report", 
                     bar_format="{l_bar}{bar:30}{r_bar}", colour='magenta') as save_progress:
                
                    with open(report_filename, 'w', encoding='utf-8') as f:
                        # Report header
                        f.write("=" * 90 + "\n")
                        f.write("ASSET CORRELATION ANALYSIS REPORT\n")
                        f.write("=" * 90 + "\n")
                        f.write(f"Analysis date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Data source: {self.data_dir}\n")
                        f.write(f"Assets in analysis: {', '.join(self.assets.keys())}\n")
                        f.write(f"Analysis period: logarithmic returns (5-minute candles)\n\n")
                        save_progress.update(20)  # 20% done
                        
                        # Results table
                        sorted_results = sorted(
                            self.results.items(),
                            key=lambda x: x[1]['correlation_abs'],
                            reverse=True
                        )
                        
                        f.write("CORRELATION ANALYSIS RESULTS\n")
                        f.write("-" * 90 + "\n")
                        f.write(f"{'№':<3} {'Pair':<12} {'Correlation':<12} {'|Corr|':<8} {'30d avg':<8} {'7d avg':<8} {'1d avg':<8} {'Stability':<15}\n")
                        f.write("-" * 90 + "\n")
                        
                        for i, (pair_name, result) in enumerate(sorted_results, 1):
                            correlation_str = f"{result['correlation']:+.4f}"
                            f.write(f"{i:<3} {result['pair']:<12} {correlation_str:<12} "
                                   f"{result['correlation_abs']:<8.4f} {result['rolling_30d_avg']:<8.4f} "
                                   f"{result['rolling_7d_avg']:<8.4f} {result['rolling_1d_avg']:<8.4f} "
                                   f"{result['correlation_stability']:<15}\n")
                        save_progress.update(30)  # 50% done
                        
                        # Detailed analysis by pairs
                        f.write("\n" + "=" * 90 + "\n")
                        f.write("DETAILED ANALYSIS BY PAIRS\n")
                        f.write("=" * 90 + "\n")
                        
                        for i, (pair_name, result) in enumerate(sorted_results, 1):
                            f.write(f"\n{i}. {result['pair']}\n")
                            f.write(f"   Correlation: {result['correlation']:+.6f}\n")
                            f.write(f"   Absolute correlation: {result['correlation_abs']:.6f}\n")
                            f.write(f"   30-day average correlation: {result['rolling_30d_avg']:+.6f}\n")
                            f.write(f"   7-day average correlation: {result['rolling_7d_avg']:+.6f}\n")
                            f.write(f"   1-day average correlation: {result['rolling_1d_avg']:+.6f}\n")
                            f.write(f"   Correlation stability: {result['correlation_stability']}\n")
                            f.write(f"   Interpretation: {self._interpret_correlation(result)}\n")
                        save_progress.update(30)  # 80% done
                        
                        # Overall statistics
                        strong_positive = sum(1 for r in self.results.values() if r['correlation'] > 0.7)
                        strong_negative = sum(1 for r in self.results.values() if r['correlation'] < -0.7)
                        moderate_positive = sum(1 for r in self.results.values() if 0.4 <= r['correlation'] <= 0.7)
                        moderate_negative = sum(1 for r in self.results.values() if -0.7 <= r['correlation'] <= -0.4)
                        weak = sum(1 for r in self.results.values() if abs(r['correlation']) < 0.4)
                        
                        f.write("\n" + "=" * 90 + "\n")
                        f.write("OVERALL STATISTICS AND CONCLUSIONS\n")
                        f.write("=" * 90 + "\n")
                        f.write(f"Total assets: {len(self.assets)}\n")
                        f.write(f"Total pairs analyzed: {len(self.results)}\n")
                        f.write(f"Strong positive correlation (> 0.7): {strong_positive}\n")
                        f.write(f"Strong negative correlation (< -0.7): {strong_negative}\n")
                        f.write(f"Moderate positive correlation (0.4 - 0.7): {moderate_positive}\n")
                        f.write(f"Moderate negative correlation (-0.7 - -0.4): {moderate_negative}\n")
                        f.write(f"Weak correlation (|r| < 0.4): {weak}\n\n")
                        
                        if sorted_results:
                            best_pair = sorted_results[0]
                            f.write("BEST PAIR FOR PAIRS ARBITRAGE:\n")
                            f.write(f"Pair: {best_pair[1]['pair']}\n")
                            f.write(f"Correlation: {best_pair[1]['correlation']:+.6f}\n")
                            f.write(f"Stability: {best_pair[1]['correlation_stability']}\n\n")
                        
                        # Strategy recommendations
                        f.write("RECOMMENDATIONS FOR pt_new.md STRATEGY:\n")
                        f.write("-" * 50 + "\n")
                        
                        excellent_pairs = [r for r in sorted_results if abs(r[1]['correlation']) > 0.8 and r[1]['correlation_stability'] in ['very_stable', 'stable']]
                        good_pairs = [r for r in sorted_results if 0.6 <= abs(r[1]['correlation']) <= 0.8 and r[1]['correlation_stability'] in ['very_stable', 'stable']]
                        
                        if excellent_pairs:
                            f.write("✅ Excellent pairs for pairs arbitrage (|r| > 0.8, stable):\n")
                            for pair_name, result in excellent_pairs:
                                f.write(f"   • {result['pair']}: r = {result['correlation']:+.4f} ({result['correlation_stability']})\n")
                        
                        if good_pairs:
                            f.write("✅ Good pairs for pairs arbitrage (0.6 ≤ |r| ≤ 0.8, stable):\n")
                            for pair_name, result in good_pairs:
                                f.write(f"   • {result['pair']}: r = {result['correlation']:+.4f} ({result['correlation_stability']})\n")
                        
                        if not excellent_pairs and not good_pairs:
                            f.write("⚠️  No strongly correlated stable pairs found.\n")
                            f.write("   Recommend collecting more data or adding other assets.\n")
                        
                        f.write(f"\nAnalysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        save_progress.update(20)  # 100% done
                
            print(f"✅ Report saved: {report_filename}")
            
        except Exception as e:
            print(f"❌ Error saving report: {e}")
    
    def _interpret_correlation(self, result):
        """Interpret correlation"""
        corr = result['correlation']
        abs_corr = result['correlation_abs']
        stability = result['correlation_stability']
        
        if abs_corr > 0.8:
            strength = "Very strong"
        elif abs_corr > 0.6:
            strength = "Strong"
        elif abs_corr > 0.4:
            strength = "Moderate"
        elif abs_corr > 0.2:
            strength = "Weak"
        else:
            strength = "Very weak"
            
        direction = "positive" if corr > 0 else "negative"
        
        if abs_corr > 0.7 and stability in ['very_stable', 'stable']:
            recommendation = "Ideal for pairs arbitrage"
        elif abs_corr > 0.5 and stability in ['very_stable', 'stable']:
            recommendation = "Good for arbitrage with caution"
        elif abs_corr > 0.3:
            recommendation = "May be suitable for arbitrage but high risk"
        else:
            recommendation = "Not recommended for pairs arbitrage"
        
        return f"{strength} {direction} correlation, stability: {stability}. {recommendation}"
    
    def create_heatmap(self):
        """Create correlation heatmap"""
        print("\n🔄 Creating heatmap...")
        
        # Progress bar for creating visualization
        with tqdm(total=100, desc="🎨 Creating visualization", 
                 bar_format="{l_bar}{bar:30}{r_bar}", colour='yellow') as viz_progress:
            
            asset_names = list(self.assets.keys())
            viz_progress.update(10)  # 10% - data preparation
            
            # Create figure with two subplots
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle('Asset Correlation Analysis (logarithmic returns)', fontsize=16, fontweight='bold')
            viz_progress.update(20)  # 30% - figure created
            
            # 1. Main correlation matrix
            sns.heatmap(self.correlation_matrix,
                       annot=True,
                       fmt='.3f',
                       xticklabels=asset_names,
                       yticklabels=asset_names,
                       cmap='RdYlBu_r',
                       center=0,
                       vmin=-1, vmax=1,
                       ax=axes[0],
                       cbar_kws={'label': 'Correlation Coefficient'})
            axes[0].set_title('Correlation Matrix\n(full data)')
            viz_progress.update(35)  # 65% - first heatmap ready
            
            # 2. Absolute correlations (for pair selection)
            abs_correlation_matrix = np.abs(self.correlation_matrix)
            np.fill_diagonal(abs_correlation_matrix, 0)  # Remove diagonal for better visualization
            
            sns.heatmap(abs_correlation_matrix,
                       annot=True,
                       fmt='.3f',
                       xticklabels=asset_names,
                       yticklabels=asset_names,
                       cmap='Reds',
                       vmin=0, vmax=1,
                       ax=axes[1],
                       cbar_kws={'label': 'Absolute Correlation'})
            axes[1].set_title('Absolute Correlations\n(for pair selection)')
            viz_progress.update(25)  # 90% - second heatmap ready
            
            plt.tight_layout()
            
            # Save to correlation folder
            filename = self.output_dir / f'correlation_heatmap_{self.analysis_timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight', 
                       facecolor='black', edgecolor='none')
            viz_progress.update(10)  # 100% - file saved
        
        print(f"✅ Heatmap saved: {filename}")
        
        # Clear memory
        plt.close()
    
    def run_full_analysis(self):
        """Run full analysis"""
        print("🚀 RUNNING CORRELATION ANALYSIS")
        print("=" * 50)
        
        try:
            # 0. Create output directory
            self.create_output_directory()
            
            # 1. Load data
            if not self.load_data():
                print("❌ Insufficient data for analysis")
                return False
                
            # 2. Synchronization
            self.sync_data()
            
            # 3. Calculate correlations
            self.calculate_correlations()
            
            # 4. Output results to terminal
            self.print_results()
            
            # 5. Save text report
            self.save_report()
            
            # 6. Create and save visualization
            self.create_heatmap()
            
            print(f"\n🎉 Analysis completed successfully!")
            print(f"📁 All results saved in folder: {self.output_dir}")
            print(f"📊 Files:")
            print(f"   • Report: correlation_report_{self.analysis_timestamp}.txt")
            print(f"   • Heatmap: correlation_heatmap_{self.analysis_timestamp}.png")
            return True
            
        except Exception as e:
            print(f"❌ Execution error: {e}")
            return False

def main():
    """Main function"""
    analyzer = CorrelationAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()
