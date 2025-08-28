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
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import r2_score
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
sns.set_palette("viridis")

class CointegrationAnalyzer:
    def __init__(self, data_dir='bybit_data', output_dir='cointegration'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.assets = {}
        self.results = {}
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
                
                # Save logarithmic prices
                log_prices = np.log(df['close'].values)
                self.assets[asset_name] = {
                    'data': df,
                    'log_prices': log_prices,
                    'prices': df['close'].values,
                    'timestamps': df['timestamp'].values if 'timestamp' in df.columns else np.arange(len(df))
                }
                
                tqdm.write(f"   ✅ {asset_name}: {len(log_prices):,} observations")
                
            except Exception as e:
                tqdm.write(f"   ❌ Error loading {file_path.name}: {e}")
                continue
        
        progress_bar.close()
        print(f"✅ Loaded assets: {', '.join(self.assets.keys())}")
        return len(self.assets) > 1
        
    def sync_data(self):
        """Time series synchronization"""
        print("\n🔄 Time series synchronization...")
        
        if len(self.assets) < 2:
            raise ValueError("Need at least 2 assets for cointegration analysis")
            
        # Find minimum length among all assets
        min_length = min(len(asset['log_prices']) for asset in self.assets.values())
        
        # Trim all series to same length (taking latest data)
        for asset_name in self.assets:
            self.assets[asset_name]['log_prices'] = self.assets[asset_name]['log_prices'][-min_length:]
            self.assets[asset_name]['prices'] = self.assets[asset_name]['prices'][-min_length:]
            
        print(f"✅ Synchronized series with length of {min_length} observations")
        
    def engle_granger_test(self, y, x, asset1, asset2):
        """
        Standard two-step Engle-Granger test
        Direction: asset1 ~ asset2 (asset1 as dependent variable)
        """
        try:
            # Step 1: Cointegration regression (without preprocessing)
            X = np.column_stack([np.ones(len(x)), x])  # Add constant
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            
            # Regression residuals
            residuals = y - (beta[0] + beta[1] * x)
            
            # R² of regression
            r2 = r2_score(y, beta[0] + beta[1] * x)
            
            # Step 2: ADF test for residual stationarity (standard maxlag formula)
            maxlag = min(int(12 * (len(residuals)/100)**(1/4)), len(residuals)//10)
            adf_result = adfuller(residuals, maxlag=maxlag, regression='c')
            
            # Single diagnostic test for autocorrelation
            ljung_box_pvalue = self._ljung_box_test(residuals)
            
            # Hedge ratio
            hedge_ratio = beta[1]
            
            # Simplified cointegration score: only ADF p-value adjusted by R²
            cointegration_score = -np.log10(max(adf_result[1], 1e-10)) * r2
            
            return {
                'adf_statistic': adf_result[0],
                'adf_pvalue': adf_result[1],
                'r_squared': r2,
                'hedge_ratio': hedge_ratio,
                'is_cointegrated': adf_result[1] < 0.05,
                'cointegration_score': cointegration_score,
                'ljung_box_pvalue': ljung_box_pvalue,
                'direction': f"{asset1} ~ {asset2}",
                'maxlag_used': maxlag
            }
            
        except Exception as e:
            print(f"   ⚠️  Error in test {asset1}-{asset2}: {e}")
            return {
                'adf_statistic': 0,
                'adf_pvalue': 1.0,
                'r_squared': 0,
                'hedge_ratio': 0,
                'is_cointegrated': False,
                'cointegration_score': 0,
                'ljung_box_pvalue': 1.0,
                'direction': f"{asset1} ~ {asset2}",
                'maxlag_used': 0
            }
    
    def _ljung_box_test(self, residuals):
        """Simplified Ljung-Box test for residual autocorrelation"""
        try:
            ljung_box = acorr_ljungbox(residuals, lags=min(10, len(residuals)//10), return_df=True)
            return ljung_box['lb_pvalue'].iloc[-1]
        except Exception:
            return 1.0  # Return neutral value if test fails
    
    def analyze_all_pairs(self):
        """Analyze cointegration of all possible asset pairs in both directions"""
        print("\n🔄 Analyzing cointegration of all asset pairs...")
        
        asset_names = list(self.assets.keys())
        pairs_list = list(combinations(asset_names, 2))
        total_tests = len(pairs_list) * 2  # Each pair tested in both directions
        
        print(f"   Total pairs: {len(pairs_list)} | Total tests: {total_tests}")
        
        # Progress bar for cointegration analysis
        progress_bar = tqdm(total=total_tests, desc="🔢 Analyzing cointegration", 
                           bar_format="{l_bar}{bar:30}{r_bar}{postfix}", colour='blue')
        
        for asset1, asset2 in pairs_list:
            y1 = self.assets[asset1]['log_prices']
            y2 = self.assets[asset2]['log_prices']
            
            # Test 1: asset1 ~ asset2 (asset1 as dependent variable)
            progress_bar.set_postfix_str(f"{asset1} ~ {asset2}")
            result1 = self.engle_granger_test(y1, y2, asset1, asset2)
            progress_bar.update(1)
            
            # Test 2: asset2 ~ asset1 (asset2 as dependent variable)  
            progress_bar.set_postfix_str(f"{asset2} ~ {asset1}")
            result2 = self.engle_granger_test(y2, y1, asset2, asset1)
            progress_bar.update(1)
            
            # Select best result from both directions
            if result1['cointegration_score'] > result2['cointegration_score']:
                best_result = result1
                dependent_asset, independent_asset = asset1, asset2
            else:
                best_result = result2
                dependent_asset, independent_asset = asset2, asset1
            
            self.results[f"{asset1}-{asset2}"] = {
                **best_result,
                'pair': f"{asset1}/{asset2}",
                'dependent_asset': dependent_asset,
                'independent_asset': independent_asset
            }
        
        progress_bar.close()
        print(f"✅ Analysis completed for {len(self.results)} pairs")
        
    def print_results(self):
        """Output results to terminal"""
        print("\n" + "="*80)
        print("📊 COINTEGRATION ANALYSIS RESULTS")
        print("="*80)
        
        # Sort by descending cointegration_score
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['cointegration_score'],
            reverse=True
        )
        
        print(f"{'№':<3} {'Pair':<12} {'Cointegr.':<10} {'ADF p-val':<10} {'R²':<8} {'Hedge Ratio':<12} {'LB p-val':<10} {'Status':<12}")
        print("-" * 95)
        
        for i, (pair_name, result) in enumerate(sorted_results, 1):
            status = "✅ COINTEGR" if result['is_cointegrated'] else "❌ NOT COINT"
            
            lb_pvalue = result.get('ljung_box_pvalue', 1.0)
            print(f"{i:<3} {result['pair']:<12} {result['cointegration_score']:<10.3f} "
                  f"{result['adf_pvalue']:<10.4f} {result['r_squared']:<8.3f} "
                  f"{result['hedge_ratio']:<12.2f} {lb_pvalue:<10.4f} {status:<12}")
                  
        # Statistics
        cointegrated_pairs = sum(1 for r in self.results.values() if r['is_cointegrated'])
        
        print("\n" + "="*80)
        print("📈 STATISTICS:")
        print(f"   • Total pairs analyzed: {len(self.results)}")
        print(f"   • Cointegrated pairs: {cointegrated_pairs}")
        print(f"   • Cointegration percentage: {cointegrated_pairs/len(self.results)*100:.1f}%")
        
        if sorted_results:
            best_pair = sorted_results[0]
            print(f"   • Best pair: {best_pair[1]['pair']} (score: {best_pair[1]['cointegration_score']:.3f})")
        
        print("="*80)
    
    def create_output_directory(self):
        """Create output directory for results"""
        try:
            self.output_dir.mkdir(exist_ok=True)
            print(f"📁 Created/found results directory: {self.output_dir}")
        except Exception as e:
            print(f"❌ Error creating directory {self.output_dir}: {e}")
            
    def save_report(self):
        """Save text report"""
        print("\n🔄 Saving text report...")
        
        try:
            # Report filename
            report_filename = self.output_dir / f'cointegration_report_{self.analysis_timestamp}.txt'
            
            # Progress bar for saving report
            with tqdm(total=100, desc="💾 Creating report", 
                     bar_format="{l_bar}{bar:30}{r_bar}", colour='magenta') as save_progress:
                
                    with open(report_filename, 'w', encoding='utf-8') as f:
                        # Report header
                        f.write("=" * 80 + "\n")
                        f.write("ASSET COINTEGRATION ANALYSIS REPORT\n")
                        f.write("=" * 80 + "\n")
                        f.write(f"Analysis date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Data source: {self.data_dir}\n")
                        f.write(f"Assets in analysis: {', '.join(self.assets.keys())}\n\n")
                        save_progress.update(20)  # 20% done
                        
                        # Results table
                        sorted_results = sorted(
                            self.results.items(),
                            key=lambda x: x[1]['cointegration_score'],
                            reverse=True
                        )
                        
                        f.write("COINTEGRATION ANALYSIS RESULTS\n")
                        f.write("-" * 80 + "\n")
                        f.write(f"{'№':<3} {'Pair':<12} {'Cointegr.':<10} {'ADF p-val':<10} {'R²':<8} {'Hedge Ratio':<12} {'LB p-val':<10} {'Status':<12}\n")
                        f.write("-" * 80 + "\n")
                        
                        for i, (pair_name, result) in enumerate(sorted_results, 1):
                            status = "✅ COINTEGR" if result['is_cointegrated'] else "❌ NOT COINT"
                            lb_pvalue = result.get('ljung_box_pvalue', 1.0)
                            f.write(f"{i:<3} {result['pair']:<12} {result['cointegration_score']:<10.3f} "
                                   f"{result['adf_pvalue']:<10.4f} {result['r_squared']:<8.3f} "
                                   f"{result['hedge_ratio']:<12.2f} {lb_pvalue:<10.4f} {status:<12}\n")
                        save_progress.update(25)  # 45% done
                        
                        # Detailed results
                        f.write("\n" + "=" * 80 + "\n")
                        f.write("DETAILED RESULTS BY PAIRS\n")
                        f.write("=" * 80 + "\n")
                        
                        for i, (pair_name, result) in enumerate(sorted_results, 1):
                            direction = result.get('direction', f"{result.get('dependent_asset', 'N/A')} ~ {result.get('independent_asset', 'N/A')}")
                            f.write(f"\n{i}. {result['pair']} ({direction})\n")
                            f.write(f"   Cointegration score: {result['cointegration_score']:.6f}\n")
                            f.write(f"   ADF statistic: {result['adf_statistic']:.6f}\n")
                            f.write(f"   ADF p-value: {result['adf_pvalue']:.6f}\n")
                            f.write(f"   R² of regression: {result['r_squared']:.6f}\n")
                            f.write(f"   Hedge ratio (β): {result['hedge_ratio']:.6f}\n")
                            f.write(f"   Ljung-Box p-value: {result.get('ljung_box_pvalue', 1.0):.6f}\n")
                            f.write(f"   Maxlag used: {result.get('maxlag_used', 'N/A')}\n")
                            f.write(f"   Cointegrated: {'Yes' if result['is_cointegrated'] else 'No'}\n")
                            f.write(f"   Interpretation: {self._interpret_result(result)}\n")
                        save_progress.update(35)  # 80% done
                        
                        # Statistics and conclusions
                        cointegrated_pairs = sum(1 for r in self.results.values() if r['is_cointegrated'])
                        f.write("\n" + "=" * 80 + "\n")
                        f.write("OVERALL STATISTICS AND CONCLUSIONS\n")
                        f.write("=" * 80 + "\n")
                        f.write(f"Total assets: {len(self.assets)}\n")
                        f.write(f"Total pairs analyzed: {len(self.results)}\n")
                        f.write(f"Cointegrated pairs: {cointegrated_pairs}\n")
                        f.write(f"Cointegration percentage: {cointegrated_pairs/len(self.results)*100:.1f}%\n\n")
                        
                        if sorted_results:
                            best_pair = sorted_results[0]
                            f.write(f"BEST PAIR FOR ARBITRAGE:\n")
                            f.write(f"Pair: {best_pair[1]['pair']}\n")
                            f.write(f"Cointegration score: {best_pair[1]['cointegration_score']:.3f}\n")
                            f.write(f"ADF p-value: {best_pair[1]['adf_pvalue']:.6f}\n")
                            f.write(f"R²: {best_pair[1]['r_squared']:.3f}\n")
                            f.write(f"Hedge ratio: {best_pair[1]['hedge_ratio']:.2f}\n\n")
                        
                        # Recommendations
                        f.write("RECOMMENDATIONS:\n")
                        f.write("-" * 40 + "\n")
                        
                        strong_pairs = [r for r in sorted_results if r[1]['cointegration_score'] > 3.0 and r[1]['is_cointegrated']]
                        if strong_pairs:
                            f.write("✅ Pairs with strong cointegration (recommended for arbitrage):\n")
                            for pair_name, result in strong_pairs:
                                f.write(f"   • {result['pair']}: score {result['cointegration_score']:.2f}\n")
                        else:
                            f.write("⚠️  No strongly cointegrated pairs found.\n")
                        
                        f.write(f"\nAnalysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        save_progress.update(20)  # 100% done
                
            print(f"✅ Report saved: {report_filename}")
            
        except Exception as e:
            print(f"❌ Error saving report: {e}")
    
    def _interpret_result(self, result):
        """Simplified interpretation of cointegration results"""
        score = result['cointegration_score']
        r2 = result['r_squared']
        pvalue = result['adf_pvalue']
        lb_pvalue = result.get('ljung_box_pvalue', 1.0)
        
        if not result['is_cointegrated']:
            return "No cointegration detected, pair not suitable for arbitrage"
        
        # Check for residual autocorrelation
        has_autocorr = lb_pvalue < 0.05
        
        # Main assessment
        if score > 5.0 and r2 > 0.8:
            base_assessment = "Excellent cointegration"
        elif score > 3.0 and r2 > 0.6:
            base_assessment = "Good cointegration"
        elif score > 1.5 and r2 > 0.4:
            base_assessment = "Weak cointegration"
        else:
            base_assessment = "Very weak cointegration"
        
        # Final recommendation
        if has_autocorr:
            if score > 3.0:
                recommendation = "suitable for arbitrage with caution (issues: residual autocorrelation)"
            else:
                recommendation = "high risk for arbitrage (issues: residual autocorrelation)"
        else:
            if score > 5.0:
                recommendation = "ideal for pairs arbitrage"
            elif score > 3.0:
                recommendation = "good for arbitrage"
            elif score > 1.5:
                recommendation = "suitable for arbitrage with caution"
            else:
                recommendation = "not recommended for arbitrage"
        
        return f"{base_assessment}, {recommendation}"
    
    def create_heatmap(self):
        """Create cointegration heatmap"""
        print("\n🔄 Creating heatmap...")
        
        # Progress bar for creating visualization
        with tqdm(total=100, desc="🎨 Creating visualization", 
                 bar_format="{l_bar}{bar:30}{r_bar}", colour='yellow') as viz_progress:
            
            asset_names = list(self.assets.keys())
            n_assets = len(asset_names)
            
            # Create matrices for different metrics
            cointegration_matrix = np.zeros((n_assets, n_assets))
            pvalue_matrix = np.ones((n_assets, n_assets))
            r2_matrix = np.zeros((n_assets, n_assets))
            viz_progress.update(15)  # 15% - matrix preparation
            
            # Fill matrices
            for pair_name, result in self.results.items():
                asset1, asset2 = result['pair'].split('/')
                i = asset_names.index(asset1)
                j = asset_names.index(asset2)
                
                # Symmetric matrix
                cointegration_matrix[i, j] = result['cointegration_score']
                cointegration_matrix[j, i] = result['cointegration_score']
                
                pvalue_matrix[i, j] = result['adf_pvalue']
                pvalue_matrix[j, i] = result['adf_pvalue']
                
                r2_matrix[i, j] = result['r_squared']
                r2_matrix[j, i] = result['r_squared']
            
            # Diagonal - self with self (maximum cointegration)
            np.fill_diagonal(cointegration_matrix, cointegration_matrix.max() * 1.1)
            np.fill_diagonal(pvalue_matrix, 0.0)
            np.fill_diagonal(r2_matrix, 1.0)
            viz_progress.update(15)  # 30% - matrices filled
            
            # Create figure with three subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Asset Cointegration Analysis', fontsize=16, fontweight='bold')
            viz_progress.update(10)  # 40% - figure created
            
            # 1. Cointegration score heatmap
            sns.heatmap(cointegration_matrix, 
                       annot=True, 
                       fmt='.2f',
                       xticklabels=asset_names,
                       yticklabels=asset_names,
                       cmap='viridis',
                       ax=axes[0],
                       cbar_kws={'label': 'Cointegration Score'})
            axes[0].set_title('Cointegration Score\n(higher is better)')
            viz_progress.update(20)  # 60% - first heatmap ready
            
            # 2. P-values heatmap (inverted color scheme)
            sns.heatmap(pvalue_matrix,
                       annot=True,
                       fmt='.3f', 
                       xticklabels=asset_names,
                       yticklabels=asset_names,
                       cmap='viridis_r',  # Inverted scheme
                       ax=axes[1],
                       cbar_kws={'label': 'ADF p-value'})
            axes[1].set_title('ADF p-value\n(< 0.05 = cointegration)')
            viz_progress.update(20)  # 80% - second heatmap ready
            
            # 3. R² heatmap
            sns.heatmap(r2_matrix,
                       annot=True,
                       fmt='.3f',
                       xticklabels=asset_names, 
                       yticklabels=asset_names,
                       cmap='viridis',
                       ax=axes[2],
                       cbar_kws={'label': 'R² Score'})
            axes[2].set_title('R² of Cointegration Regression\n(goodness of fit)')
            viz_progress.update(15)  # 95% - third heatmap ready
            
            plt.tight_layout()
            
            # Save to cointegration folder
            filename = self.output_dir / f'cointegration_heatmap_{self.analysis_timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight', 
                       facecolor='black', edgecolor='none')
            viz_progress.update(5)  # 100% - file saved
        
        print(f"✅ Heatmap saved: {filename}")
        
        # Show plot (optional)
        # plt.show()
        
        # Clear memory
        plt.close()
        
    def run_full_analysis(self):
        """Run full analysis"""
        print("🚀 RUNNING COINTEGRATION ANALYSIS")
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
            
            # 3. Cointegration analysis
            self.analyze_all_pairs()
            
            # 4. Output results to terminal
            self.print_results()
            
            # 5. Save text report
            self.save_report()
            
            # 6. Create and save visualization
            self.create_heatmap()
            
            print(f"\n🎉 Analysis completed successfully!")
            print(f"📁 All results saved in folder: {self.output_dir}")
            print(f"📊 Files:")
            print(f"   • Report: cointegration_report_{self.analysis_timestamp}.txt")
            print(f"   • Heatmap: cointegration_heatmap_{self.analysis_timestamp}.png")
            return True
            
        except Exception as e:
            print(f"❌ Execution error: {e}")
            return False

def main():
    """Main function"""
    analyzer = CointegrationAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()
