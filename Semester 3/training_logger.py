#!/usr/bin/env python
# coding: utf-8

import os
import json
import csv
import numpy as np
from datetime import datetime

class TrainingLogger:
    """
    Comprehensive training logger that saves all training data for later analysis.
    Saves data in both JSON (for easy loading) and CSV (for Excel/pandas analysis).
    """
    
    def __init__(self, config):
        self.config = config
        self.log_dir = os.path.join(config.model_dir, 'training_logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize data storage
        self.training_data = {
            'config': vars(config),
            'training_started': datetime.now().isoformat(),
            'episodes': {},  # Will store per-episode data
            'test_results': {},  # Will store periodic test results
            'phase_analysis': {},  # Will store phase-specific results
            'final_results': {}  # Will store final evaluation results
        }
        
        # Per-episode metrics storage
        self.episode_buffer = []
        self.test_buffer = []
        
    def log_episode(self, agent_type, episode_num, reward, demand_sequence, 
                   disruption_info=None, warning_sequence=None):
        """Log data from a single training episode."""
        episode_data = {
            'agent_type': agent_type,
            'episode': episode_num,
            'cumulative_reward': float(reward),
            'demand_mean': float(np.mean(demand_sequence)),
            'demand_std': float(np.std(demand_sequence)),
            'timestamp': datetime.now().isoformat()
        }
        
        if disruption_info:
            episode_data.update({
                'disruption_start': int(disruption_info['start']),
                'disruption_duration': int(disruption_info['duration'])
            })
            
        if warning_sequence is not None:
            episode_data['warning_count'] = int(np.sum(warning_sequence))
            
        self.episode_buffer.append(episode_data)
        
        # Save periodically to avoid memory issues
        if len(self.episode_buffer) >= 1000:
            self._flush_episode_buffer(agent_type)
    
    def log_test_results(self, agent_type, episode_num, test_results, 
                        disruption_info=None):
        """Log results from periodic testing."""
        test_data = {
            'agent_type': agent_type,
            'episode': episode_num,
            'timestamp': datetime.now().isoformat(),
            'avg_kpis': test_results.get('avg_kpi', {}),
            'ci_kpis': test_results.get('ci_kpis', {}),
            'phase_kpis': test_results.get('phase_kpis', {})
        }
        
        # Flatten KPIs for easier analysis
        for kpi in ['BW', 'RE', 'BRI', 'Cost']:
            test_data[f'{kpi}_mean'] = test_results.get('avg_kpi', {}).get(kpi, np.nan)
            ci_data = test_results.get('ci_kpis', {}).get(kpi, {})
            test_data[f'{kpi}_ci_lower'] = ci_data.get('ci_lower', np.nan)
            test_data[f'{kpi}_ci_upper'] = ci_data.get('ci_upper', np.nan)
        
        if disruption_info:
            test_data['disruption_info'] = disruption_info
            
        self.test_buffer.append(test_data)
        
        # Also save immediately for test results
        self._flush_test_buffer(agent_type)
        
    def log_final_evaluation(self, agent_type, final_results):
        """Log final evaluation results."""
        self.training_data['final_results'][agent_type] = {
            'kpis': final_results.get('avg_kpi', {}),
            'confidence_intervals': final_results.get('ci_kpis', {}),
            'phase_analysis': final_results.get('phase_kpis', {}),
            'evaluation_time': datetime.now().isoformat()
        }
        self.save_all()
        
    def _flush_episode_buffer(self, agent_type):
        """Save episode buffer to CSV."""
        if not self.episode_buffer:
            return
            
        filename = f"episodes_{agent_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(self.log_dir, filename)
        
        # Save without pandas to avoid dependency
        import csv
        with open(filepath, 'w', newline='') as f:
            if self.episode_buffer:
                writer = csv.DictWriter(f, fieldnames=self.episode_buffer[0].keys())
                writer.writeheader()
                writer.writerows(self.episode_buffer)
        
        print(f"Saved {len(self.episode_buffer)} episode records to {filename}")
        self.episode_buffer = []
        
    def _flush_test_buffer(self, agent_type):
        """Save test results to JSON."""
        if not self.test_buffer:
            return
            
        filename = f"test_results_{agent_type}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        # Load existing data if file exists
        existing_data = []
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                existing_data = json.load(f)
                
        # Append new data
        existing_data.extend(self.test_buffer)
        
        # Save updated data
        with open(filepath, 'w') as f:
            json.dump(existing_data, f, indent=2)
            
        print(f"Saved {len(self.test_buffer)} test results to {filename}")
        self.test_buffer = []
        
    def save_all(self):
        """Save all training data to master JSON file."""
        # Flush any remaining buffers
        for agent_type in ['Strm', 'srdqn', 'ewa']:
            self._flush_episode_buffer(agent_type)
            self._flush_test_buffer(agent_type)
            
        # Save master training data
        master_file = os.path.join(self.log_dir, 'training_master.json')
        with open(master_file, 'w') as f:
            json.dump(self.training_data, f, indent=2)
            
        print(f"Saved master training data to {master_file}")
        
    def create_summary_report(self):
        """Create a summary CSV with key metrics for easy analysis."""
        summary_data = []
        
        # Collect final results for each agent
        for agent_type, results in self.training_data['final_results'].items():
            row = {
                'agent': agent_type,
                'final_cost': results['kpis'].get('Cost', np.nan),
                'final_bw': results['kpis'].get('BW', np.nan),
                'final_re': results['kpis'].get('RE', np.nan),
                'final_bri': results['kpis'].get('BRI', np.nan)
            }
            
            # Add confidence intervals
            for kpi in ['Cost', 'BW', 'RE', 'BRI']:
                ci_data = results['confidence_intervals'].get(kpi, {})
                row[f'{kpi}_ci_lower'] = ci_data.get('ci_lower', np.nan)
                row[f'{kpi}_ci_upper'] = ci_data.get('ci_upper', np.nan)
                
            # Add phase-specific costs
            for phase in ['pre', 'during', 'post']:
                phase_data = results['phase_analysis'].get(phase, {})
                row[f'{phase}_cost'] = phase_data.get('Cost', np.nan)
                row[f'{phase}_bw'] = phase_data.get('BW', np.nan)
                
            summary_data.append(row)
            
        # Save summary without pandas
        if summary_data:
            summary_file = os.path.join(self.log_dir, 'results_summary.csv')
            with open(summary_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
                writer.writeheader()
                writer.writerows(summary_data)
            print(f"Created summary report: {summary_file}")