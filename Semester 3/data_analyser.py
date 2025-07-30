#!/usr/bin/env python
# coding: utf-8

"""
Combined Training and CI Analyzer
=================================
Combines training data extraction and confidence interval analysis into one script.
Creates comprehensive visualizations without duplicating figures.

Usage:
    python combined_analyzer.py --log_dir "logs/run_YYYYMMDD_HHMMSS"
"""

import os
import json
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from collections import defaultdict

# Set matplotlib style
plt.style.use('default')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11
})

class CombinedAnalyzer:
    """Combined analyzer for training data and confidence intervals."""
    
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.training_logs_dir = os.path.join(log_dir, 'training_logs')
        self.figures_dir = os.path.join(log_dir, 'combined_analysis')
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Agent mapping
        self.agent_labels = {
            'Strm': 'Sterman Heuristic',
            'srdqn': 'Standard DQN',
            'ewa_pd90_pf1': 'EWA-DQN (High Accuracy)',
            'ewa_pd80_pf5': 'EWA-DQN (Medium Accuracy)', 
            'ewa_pd70_pf10': 'EWA-DQN (Low Accuracy)'
        }
        
        # Clean labels for CI plots
        self.agent_labels_clean = {
            'Strm': 'Sterman\nHeuristic',
            'srdqn': 'Standard\nDQN',
            'ewa_pd90_pf1': 'EWA-DQN\n(High Acc.)',
            'ewa_pd80_pf5': 'EWA-DQN\n(Med. Acc.)', 
            'ewa_pd70_pf10': 'EWA-DQN\n(Low Acc.)'
        }
        
        # Colors
        self.colors = {
            'Strm': '#1f77b4',
            'srdqn': '#ff7f0e', 
            'ewa_pd90_pf1': '#2ca02c',
            'ewa_pd80_pf5': '#d62728',
            'ewa_pd70_pf10': '#9467bd'
        }
        
        # Load all data
        self.test_data = self._load_test_data()
        self.master_data = self._load_master_data()
        self.ci_data = self._load_ci_data()
        
    def _load_test_data(self):
        """Load test results from JSON files."""
        test_files = glob.glob(os.path.join(self.training_logs_dir, 'test_results_*.json'))
        all_tests = {}
        
        for file in test_files:
            agent_type = os.path.basename(file).replace('test_results_', '').replace('.json', '')
            try:
                with open(file, 'r') as f:
                    all_tests[agent_type] = json.load(f)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                
        return all_tests
    
    def _load_master_data(self):
        """Load master training data."""
        master_file = os.path.join(self.training_logs_dir, 'training_master.json')
        if os.path.exists(master_file):
            with open(master_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_ci_data(self):
        """Load confidence interval data from results_summary.csv files."""
        summary_files = []
        
        # Look for results_summary.csv files
        main_file = os.path.join(self.log_dir, 'results_summary.csv')
        if os.path.exists(main_file):
            summary_files.append(main_file)
            
        training_file = os.path.join(self.training_logs_dir, 'results_summary.csv')
        if os.path.exists(training_file):
            summary_files.append(training_file)
        
        # Search recursively
        pattern = os.path.join(self.log_dir, '**', 'results_summary.csv')
        found_files = glob.glob(pattern, recursive=True)
        summary_files.extend([f for f in found_files if f not in summary_files])
        
        all_data = []
        for file_path in summary_files:
            try:
                print(f"Loading CI data: {os.path.relpath(file_path, self.log_dir)}")
                with open(file_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        processed_row = {'agent': row['agent']}
                        # Convert numeric columns
                        for key, value in row.items():
                            if key != 'agent':
                                try:
                                    processed_row[key] = float(value) if value and value.lower() != 'nan' else np.nan
                                except:
                                    processed_row[key] = np.nan
                        all_data.append(processed_row)
                        
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return all_data
    
    def extract_training_progress(self):
        """Extract training progress from test data."""
        training_progress = {}
        
        for agent_type, test_results in self.test_data.items():
            if not test_results:
                continue
                
            episodes = []
            costs = []
            bw_ratios = []
            
            for result in test_results:
                episodes.append(result.get('episode', 0))
                costs.append(result.get('Cost_mean', np.nan))
                bw_ratios.append(result.get('BW_mean', np.nan))
            
            training_progress[agent_type] = {
                'episodes': np.array(episodes),
                'costs': np.array(costs),
                'bw_ratios': np.array(bw_ratios)
            }
        
        return training_progress
    
    def plot_training_progress(self):
        """Create training progress visualization."""
        progress_data = self.extract_training_progress()
        
        if not progress_data:
            print("No training progress data found.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Cost progression
        for agent_type, data in progress_data.items():
            if agent_type in self.agent_labels and len(data['episodes']) > 0:
                label = self.agent_labels[agent_type]
                color = self.colors.get(agent_type, '#333333')
                
                # Filter out NaN values
                valid_idx = ~np.isnan(data['costs'])
                if np.sum(valid_idx) > 0:
                    episodes_clean = data['episodes'][valid_idx]
                    costs_clean = data['costs'][valid_idx]
                    
                    ax1.plot(episodes_clean, costs_clean, 'o-', 
                            color=color, label=label, markersize=4, linewidth=2)
        
        ax1.set_xlabel('Training Episodes')
        ax1.set_ylabel('Average Total Cost')
        ax1.set_title('Cost Reduction During Training')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Bullwhip progression
        for agent_type, data in progress_data.items():
            if agent_type in self.agent_labels and len(data['episodes']) > 0:
                label = self.agent_labels[agent_type]
                color = self.colors.get(agent_type, '#333333')
                
                valid_idx = ~np.isnan(data['bw_ratios'])
                if np.sum(valid_idx) > 0:
                    episodes_clean = data['episodes'][valid_idx]
                    bw_clean = data['bw_ratios'][valid_idx]
                    
                    ax2.plot(episodes_clean, bw_clean, 'o-',
                            color=color, label=label, markersize=4, linewidth=2)
        
        ax2.set_xlabel('Training Episodes')
        ax2.set_ylabel('Bullwhip Ratio')
        ax2.set_title('Bullwhip Reduction During Training')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = os.path.join(self.figures_dir, 'training_progress.pdf')
        plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Training progress saved: {output_file}")
        plt.close()
    
    def plot_ewa_sensitivity(self):
        """Create EWA sensitivity analysis."""
        ewa_agents = [k for k in self.test_data.keys() if k.startswith('ewa_')]
        
        if len(ewa_agents) < 2:
            print("Insufficient EWA agent data for sensitivity analysis.")
            return
        
        # Extract sensitivity data
        sensitivity_data = []
        for agent in ewa_agents:
            # Parse pd and pf from agent name
            parts = agent.split('_')
            if len(parts) >= 3:
                try:
                    pd = int(parts[1][2:]) / 100.0
                    pf = int(parts[2][2:]) / 100.0
                    accuracy = pd - pf
                    
                    # Get final performance
                    if agent in self.test_data and self.test_data[agent]:
                        final_result = self.test_data[agent][-1]
                        sensitivity_data.append({
                            'pd': pd,
                            'pf': pf,
                            'accuracy': accuracy,
                            'cost': final_result.get('Cost_mean', np.nan),
                            'bw': final_result.get('BW_mean', np.nan),
                            'agent': agent
                        })
                except (ValueError, IndexError):
                    continue
        
        if len(sensitivity_data) < 2:
            print("Not enough EWA sensitivity data.")
            return
        
        # Extract arrays for plotting
        accuracies = [d['accuracy'] for d in sensitivity_data]
        costs = [d['cost'] for d in sensitivity_data]
        bw_ratios = [d['bw'] for d in sensitivity_data]
        pf_values = [d['pf'] for d in sensitivity_data]
        pd_values = [d['pd'] for d in sensitivity_data]
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Cost vs Accuracy
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(accuracies, costs, c=pf_values, s=100, cmap='viridis', alpha=0.7)
        ax1.set_xlabel('Signal Accuracy (pd - pf)')
        ax1.set_ylabel('Total Cost')
        ax1.set_title('Cost vs Early Warning Accuracy')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='False Positive Rate')
        
        # Plot 2: Bullwhip vs Accuracy  
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(accuracies, bw_ratios, c=pf_values, s=100, cmap='viridis', alpha=0.7)
        ax2.set_xlabel('Signal Accuracy (pd - pf)')
        ax2.set_ylabel('Bullwhip Ratio')
        ax2.set_title('Bullwhip vs Early Warning Accuracy')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='False Positive Rate')
        
        # Plot 3: Performance Map
        ax3 = axes[1, 0]
        scatter3 = ax3.scatter(pd_values, pf_values, c=costs, s=200, cmap='RdYlBu_r', alpha=0.8)
        ax3.set_xlabel('True Positive Rate (pd)')
        ax3.set_ylabel('False Positive Rate (pf)')
        ax3.set_title('Cost Performance Map')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=ax3, label='Total Cost')
        
        # Plot 4: Signal Quality
        ax4 = axes[1, 1]
        signal_quality = [pd_val / (pd_val + pf_val) if (pd_val + pf_val) > 0 else 0 
                         for pd_val, pf_val in zip(pd_values, pf_values)]
        scatter4 = ax4.scatter(signal_quality, costs, c=accuracies, s=100, cmap='plasma', alpha=0.7)
        ax4.set_xlabel('Signal Quality (pd/(pd+pf))')
        ax4.set_ylabel('Total Cost')
        ax4.set_title('Performance vs Signal Quality')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter4, ax=ax4, label='Accuracy')
        
        plt.tight_layout()
        
        output_file = os.path.join(self.figures_dir, 'ewa_sensitivity_analysis.pdf')
        plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
        print(f"EWA sensitivity analysis saved: {output_file}")
        plt.close()
    
    def plot_performance_comparison(self):
        """Create performance comparison using master data (NOT CI data to avoid duplication)."""
        if not self.master_data or 'final_results' not in self.master_data:
            print("No master results data available for performance comparison.")
            return
        
        final_results = self.master_data['final_results']
        
        # Extract data
        agents = []
        costs = []
        bw_ratios = []
        colors_list = []
        
        for agent_type, results in final_results.items():
            if agent_type in self.agent_labels:
                agents.append(self.agent_labels[agent_type])
                colors_list.append(self.colors.get(agent_type, '#333333'))
                
                kpis = results.get('kpis', {})
                costs.append(kpis.get('Cost', 0))
                bw_ratios.append(kpis.get('BW', 0))
        
        if not agents:
            print("No performance comparison data available.")
            return
        
        # Create comparison plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Cost comparison
        bars1 = ax1.bar(range(len(agents)), costs, color=colors_list, alpha=0.7)
        ax1.set_xticks(range(len(agents)))
        ax1.set_xticklabels([name.replace(' ', '\n') for name in agents], fontsize=9)
        ax1.set_ylabel('Total Cost')
        ax1.set_title('Performance Comparison - Total Cost')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, cost in zip(bars1, costs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{cost:.0f}', ha='center', va='bottom')
        
        # Plot 2: Bullwhip comparison
        bars2 = ax2.bar(range(len(agents)), bw_ratios, color=colors_list, alpha=0.7)
        ax2.set_xticks(range(len(agents)))
        ax2.set_xticklabels([name.replace(' ', '\n') for name in agents], fontsize=9)
        ax2.set_ylabel('Bullwhip Ratio')
        ax2.set_title('Performance Comparison - Bullwhip Ratio')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, bw in zip(bars2, bw_ratios):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{bw:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        output_file = os.path.join(self.figures_dir, 'performance_comparison.pdf')
        plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Performance comparison saved: {output_file}")
        plt.close()
    
    def create_ci_summary_table(self):
        """Create confidence interval summary table."""
        if not self.ci_data:
            print("No CI data available for summary table.")
            return
        
        # Sort by cost (best to worst)
        sorted_data = sorted(self.ci_data, key=lambda x: x.get('final_cost', float('inf')))
        
        # Create summary file
        summary_file = os.path.join(self.figures_dir, 'performance_summary_with_ci.txt')
        
        with open(summary_file, 'w') as f:
            f.write("PERFORMANCE SUMMARY WITH 95% CONFIDENCE INTERVALS\n")
            f.write("=" * 60 + "\n")
            
            for i, row in enumerate(sorted_data, 1):
                agent = row['agent']
                if agent not in self.agent_labels_clean:
                    continue
                
                f.write(f"{i}. {self.agent_labels_clean[agent].replace(chr(10), ' ')}\n")
                f.write("-" * 30 + "\n")
                
                # Total Cost
                cost = row.get('final_cost', np.nan)
                cost_lower = row.get('Cost_ci_lower', np.nan)
                cost_upper = row.get('Cost_ci_upper', np.nan)
                if not np.isnan(cost):
                    if not np.isnan(cost_lower) and not np.isnan(cost_upper):
                        f.write(f"Total Cost: {cost:.0f} (95% CI: {cost_lower:.0f} - {cost_upper:.0f})\n")
                    else:
                        f.write(f"Total Cost: {cost:.0f} (CI not available)\n")
                
                # Bullwhip Ratio
                bw = row.get('final_bw', np.nan)
                bw_lower = row.get('BW_ci_lower', np.nan)
                bw_upper = row.get('BW_ci_upper', np.nan)
                if not np.isnan(bw):
                    if not np.isnan(bw_lower) and not np.isnan(bw_upper):
                        f.write(f"Bullwhip Ratio: {bw:.2f} (95% CI: {bw_lower:.2f} - {bw_upper:.2f})\n")
                    else:
                        f.write(f"Bullwhip Ratio: {bw:.2f} (CI not available)\n")
                
                # Ripple Effect
                re = row.get('final_re', np.nan)
                re_lower = row.get('RE_ci_lower', np.nan)
                re_upper = row.get('RE_ci_upper', np.nan)
                if not np.isnan(re):
                    if not np.isnan(re_lower) and not np.isnan(re_upper):
                        f.write(f"Ripple Effect: {re:.2f} (95% CI: {re_lower:.2f} - {re_upper:.2f})\n")
                    else:
                        f.write(f"Ripple Effect: {re:.2f} (CI not available)\n")
                
                f.write("\n")
            
            f.write("NOTES:\n")
            f.write("- 95% confidence intervals based on multiple test evaluations\n")
            f.write("- Lower values are better for Cost and Bullwhip Ratio\n")
            f.write("- Ripple Effect close to 1.0 indicates good supply chain stability\n")
        
        print(f"📄 CI summary table saved: {summary_file}")
        
        # Also print to console
        print(f"\n" + "="*60)
        with open(summary_file, 'r') as f:
            print(f.read())
    
    def plot_ci_ranking(self):
        """Create clean confidence interval ranking plot."""
        if not self.ci_data:
            print("No CI data available for ranking plot.")
            return
        
        # Prepare data - focus on cost (most important metric)
        agents_data = []
        for row in self.ci_data:
            agent = row['agent']
            if agent in self.agent_labels_clean and not np.isnan(row.get('final_cost', np.nan)):
                agents_data.append({
                    'agent': agent,
                    'label': self.agent_labels_clean[agent],
                    'cost': row['final_cost'],
                    'cost_lower': row.get('Cost_ci_lower', row['final_cost']),
                    'cost_upper': row.get('Cost_ci_upper', row['final_cost']),
                    'color': self.colors.get(agent, '#666666')
                })
        
        if len(agents_data) < 2:
            print("❌ Need at least 2 agents for CI ranking plot")
            return
        
        # Sort by performance (best to worst)
        agents_data.sort(key=lambda x: x['cost'])
        
        # Create clean horizontal plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_positions = np.arange(len(agents_data))
        
        for i, agent_info in enumerate(agents_data):
            cost = agent_info['cost']
            ci_lower = agent_info['cost_lower']
            ci_upper = agent_info['cost_upper']
            color = agent_info['color']
            
            # Calculate error bars
            error_lower = cost - ci_lower if not np.isnan(ci_lower) else 0
            error_upper = ci_upper - cost if not np.isnan(ci_upper) else 0
            
            # Plot point with error bar - simple circles only
            ax.errorbar(cost, i, xerr=[[error_lower], [error_upper]], 
                       fmt='o', color=color, markersize=12, linewidth=3,
                       capsize=8, capthick=3, alpha=0.8)
            
            # Add clear value labels
            if not np.isnan(ci_lower) and not np.isnan(ci_upper):
                # Lower bound label
                ax.text(ci_lower, i + 0.15, f'{ci_lower:.0f}', 
                       ha='center', va='bottom', fontsize=10, 
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.7))
                
                # Upper bound label  
                ax.text(ci_upper, i + 0.15, f'{ci_upper:.0f}', 
                       ha='center', va='bottom', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='lightcoral', alpha=0.7))
                
                # Mean value label
                ax.text(cost, i - 0.25, f'{cost:.0f}', 
                       ha='center', va='top', fontsize=11, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        
        # Customize plot
        ax.set_yticks(y_positions)
        ax.set_yticklabels([a['label'] for a in agents_data])
        ax.set_xlabel('Total Cost')
        ax.set_title('Performance Ranking with 95% Confidence Intervals', fontsize=16, pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add legend for CI bounds
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', alpha=0.7, label='Lower 95% CI'),
            Patch(facecolor='lightcoral', alpha=0.7, label='Upper 95% CI'),
            Patch(facecolor='white', alpha=0.9, label='Mean Value')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        # Set reasonable x-axis limits
        all_costs = [a['cost'] for a in agents_data]
        all_lowers = [a['cost_lower'] for a in agents_data if not np.isnan(a['cost_lower'])]
        all_uppers = [a['cost_upper'] for a in agents_data if not np.isnan(a['cost_upper'])]
        
        if all_lowers and all_uppers:
            margin = (max(all_uppers) - min(all_lowers)) * 0.1
            ax.set_xlim(min(all_lowers) - margin, max(all_uppers) + margin)
        
        plt.tight_layout()
        
        # Save
        output_file = os.path.join(self.figures_dir, 'performance_ranking_with_ci.pdf')
        plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
        print(f"📊 CI ranking plot saved: {output_file}")
        plt.close()
    
    def create_training_summary(self):
        """Create training summary statistics."""
        progress_data = self.extract_training_progress()
        
        summary = {}
        for agent_type, data in progress_data.items():
            if agent_type in self.agent_labels and len(data['costs']) > 0:
                valid_costs = data['costs'][~np.isnan(data['costs'])]
                if len(valid_costs) > 0:
                    summary[agent_type] = {
                        'agent_name': self.agent_labels[agent_type],
                        'total_episodes': int(data['episodes'][-1]) if len(data['episodes']) > 0 else 0,
                        'initial_cost': float(valid_costs[0]),
                        'final_cost': float(valid_costs[-1]),
                        'best_cost': float(np.min(valid_costs)),
                        'improvement': float(valid_costs[0] - valid_costs[-1]),
                        'volatility': float(np.std(valid_costs) / np.mean(valid_costs)) if np.mean(valid_costs) > 0 else 0
                    }
        
        # Save summary
        summary_file = os.path.join(self.figures_dir, 'training_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Training summary saved: {summary_file}")
        
        return summary
    
    def run_analysis(self):
        """Run complete combined analysis."""
        print(f"\n{'='*60}")
        print(f"COMBINED TRAINING AND CI ANALYSIS")
        print(f"{'='*60}")
        
        # Check data availability
        if self.test_data:
            total_tests = sum(len(tests) for tests in self.test_data.values())
            print(f"✅ Training data: {total_tests} test records for {len(self.test_data)} agents")
        else:
            print("⚠️  No training test data found")
        
        if self.ci_data:
            print(f"✅ CI data: {len(self.ci_data)} agent records")
        else:
            print("⚠️  No confidence interval data found")
        
        if self.master_data:
            print(f"✅ Master data: Available")
        else:
            print("⚠️  No master data found")
        
        # Generate all visualizations (avoiding duplicates)
        print(f"\nGenerating visualizations...")
        
        # Training-specific plots
        self.plot_training_progress()
        self.plot_ewa_sensitivity()
        
        # Performance comparison (from master data - not CI data)
        self.plot_performance_comparison()
        
        # CI-specific outputs (only if CI data available)
        if self.ci_data:
            self.create_ci_summary_table()
            self.plot_ci_ranking()
        
        # Training summary
        summary = self.create_training_summary()
        
        # Print training summary
        if summary:
            print(f"\nTRAINING SUMMARY:")
            print(f"{'-'*40}")
            for agent_type, stats in summary.items():
                print(f"\n{stats['agent_name']}:")
                print(f"  Episodes: {stats['total_episodes']:,}")
                print(f"  Initial→Final Cost: {stats['initial_cost']:.0f} → {stats['final_cost']:.0f}")
                print(f"  Improvement: {stats['improvement']:.0f}")
                print(f"  Best Cost: {stats['best_cost']:.0f}")
        
        print(f"\n🎯 COMBINED ANALYSIS COMPLETE!")
        print(f"📁 Results saved to: {self.figures_dir}")
        print(f"📊 Generated files:")
        print(f"   • training_progress.pdf")
        print(f"   • ewa_sensitivity_analysis.pdf") 
        print(f"   • performance_comparison.pdf")
        if self.ci_data:
            print(f"   • performance_summary_with_ci.txt")
            print(f"   • performance_ranking_with_ci.pdf")
        print(f"   • training_summary.json")

def find_latest_log_dir(base_log_dir='logs'):
    """Find the most recent log directory."""
    if not os.path.exists(base_log_dir):
        return None
    
    pattern = os.path.join(base_log_dir, 'run_*')
    run_dirs = glob.glob(pattern)
    
    if not run_dirs:
        return None
    
    run_dirs.sort(key=os.path.getmtime, reverse=True)
    return run_dirs[0]

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Combined training and CI analysis')
    parser.add_argument('--log_dir', help='Path to log directory')
    args = parser.parse_args()
    
    if args.log_dir:
        log_dir = args.log_dir
    else:
        log_dir = find_latest_log_dir()
        if not log_dir:
            print("❌ No log directories found. Use --log_dir to specify path.")
            return
        print(f"Using latest log directory: {log_dir}")
    
    if not os.path.exists(log_dir):
        print(f"❌ Directory not found: {log_dir}")
        return
    
    analyzer = CombinedAnalyzer(log_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()