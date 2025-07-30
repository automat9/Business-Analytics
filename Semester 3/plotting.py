#!/usr/bin/env python
# coding: utf-8
# Plots created immediately after training is finished

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

def _ewa_keys(container):
    """Return every key that corresponds to an EWA run."""
    return [k for k in container.keys() if k.startswith('ewa_pd')]

def _label_from_key(agent_key):
    """Turn e.g. 'ewa_pd90_pf1' into 'EWA-DQN (pd=0.90, pf=0.01)'."""
    if agent_key == 'Strm':
        return 'Sterman'
    if agent_key == 'srdqn':
        return 'Standard DQN'
    if agent_key.startswith('ewa_pd'):
        parts = agent_key.split('_')
        pd = int(parts[1][2:]) / 100.0
        pf = int(parts[2][2:]) / 100.0
        return f'EWA-DQN (pd={pd:.2f}, pf={pf:.2f})'
    return agent_key

def _color_map(keys):
    """Give each key its own color using a matplotlib categorical palette."""
    # Define specific colors for consistency
    color_dict = {
        'Strm': '#1f77b4',      # blue
        'srdqn': '#ff7f0e',     # orange
        'ewa_pd90_pf1': '#2ca02c',   # green - high accuracy
        'ewa_pd80_pf5': '#d62728',   # red - medium accuracy  
        'ewa_pd70_pf10': '#9467bd',  # purple - low accuracy
    }
    
    # For any keys not in the predefined dict, assign colors from tab10
    cmap = plt.get_cmap('tab10')
    all_keys = sorted(keys)
    for i, k in enumerate(all_keys):
        if k not in color_dict:
            color_dict[k] = cmap(i + 5)  # Start from index 5 to avoid duplicates
    
    return color_dict

def plot_training_progress(training_history, config):
    """Plot training progress for all agents with dynamic EWA handling."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Get all agent keys
    all_keys = list(training_history.keys())
    colors = _color_map(all_keys)

    # Plot each agent
    for key in all_keys:
        if key in training_history and 'episodes' in training_history[key]:
            label = _label_from_key(key)
            style = '-' if key in ['Strm', 'srdqn'] else '--'
            
            ax1.plot(training_history[key]['episodes'],
                     training_history[key]['costs'],
                     style, linewidth=2, color=colors[key],
                     label=label)
            ax2.plot(training_history[key]['episodes'],
                     training_history[key]['bw'],
                     style, linewidth=2, color=colors[key],
                     label=label)

    ax1.set_xlabel('Training Episodes')
    ax1.set_ylabel('Average Total Cost')
    ax1.set_title('Cost Reduction During Training')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(bottom=0)  # Ensure y-axis starts at 0

    ax2.set_xlabel('Training Episodes')
    ax2.set_ylabel('Average Bullwhip Ratio')
    ax2.set_title('Bullwhip Reduction During Training')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(bottom=0)  # Ensure y-axis starts at 0

    plt.tight_layout()
    outdir = os.path.join(config.model_dir, 'saved_figures')
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, 'training_progress.pdf')
    plt.savefig(fname, format='pdf', dpi=300)
    print(f"Training progress plot saved: {fname}")
    plt.close()


def plot_kpi_comparison(kpi_results, config):
    """
    Bar chart comparing KPIs across all agents.
    """
    agents = list(kpi_results.keys())
    kpis = ['BW', 'RE', 'BRI', 'Cost']
    
    # Create figure with subplots for each KPI
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    colors = _color_map(agents)
    
    for idx, kpi in enumerate(kpis):
        ax = axes[idx]
        
        # Get values for this KPI
        values = []
        labels = []
        bar_colors = []
        
        for agent in agents:
            if kpi in kpi_results[agent]:
                values.append(kpi_results[agent][kpi])
                labels.append(_label_from_key(agent).replace(' ', '\n'))
                bar_colors.append(colors.get(agent, '#333333'))
        
        # Create bar plot
        bars = ax.bar(range(len(values)), values, color=bar_colors)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=0, fontsize=8)
        ax.set_ylabel(kpi)
        ax.set_title(f'{kpi} Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if not np.isnan(value) and not np.isinf(value):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}', ha='center', va='bottom')
    
    plt.suptitle('KPI Comparison Across All Agents', fontsize=14)
    plt.tight_layout()
    
    # Save
    outdir = os.path.join(config.model_dir, 'saved_figures')
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, 'kpi_comparison.pdf')
    plt.savefig(fname, format='pdf', dpi=300)
    print(f"KPI comparison plot saved: {fname}")
    plt.close()

def plot_inventory_vs_demand(histories, disruption_info, config):
    """
    Time series plot showing inventory levels vs demand for all agents.
    Highlights the disruption period.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get color scheme
    agents = list(histories.keys())
    colors = _color_map(agents)
    
    # Plot demand (same for all agents)
    first_agent = list(histories.keys())[0]
    demand = histories[first_agent]['demand']
    time = range(len(demand))
    ax.plot(time, demand, 'k--', label='Customer Demand', linewidth=2, alpha=0.7)
    
    # Plot inventory levels for each agent
    for agent, history in histories.items():
        inventory = history['inventory_levels']
        label = _label_from_key(agent)
        ax.plot(time, inventory, label=label, linewidth=2, 
                color=colors.get(agent, '#333333'))
    
    # Highlight disruption period
    if disruption_info:
        start = disruption_info['start']
        duration = disruption_info['duration']
        ax.axvspan(start, start + duration - 1, alpha=0.2, color='red', 
                  label='Disruption Period')
        
        # Also mark warning period (2 periods before)
        if start >= 2:
            ax.axvspan(start - 2, start - 1, alpha=0.1, color='orange', 
                      label='Warning Window')
    
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Units')
    ax.set_title('Inventory Levels vs Customer Demand Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_xlim(0, len(demand)-1)
    
    # Add zero line
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    
    # Save
    outdir = os.path.join(config.model_dir, 'saved_figures')
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, 'inventory_vs_demand.pdf')
    plt.savefig(fname, format='pdf', dpi=300)
    print(f"Inventory vs demand plot saved: {fname}")
    plt.close()

def plot_phase_analysis(phase_kpis, config):
    """
    Plot KPIs broken down by phase (pre/during/post disruption) for RQ2.
    """
    agents = list(phase_kpis.keys())
    phases = ['pre', 'during', 'post']
    kpis = ['BW', 'Cost']  # Focus on key metrics for phase analysis
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = _color_map(agents)
    
    for idx, kpi in enumerate(kpis):
        ax = axes[idx]
        
        # Prepare data for grouped bar chart
        x = np.arange(len(phases))
        width = 0.15
        
        for i, agent in enumerate(agents):
            values = []
            for phase in phases:
                val = phase_kpis[agent][phase][kpi]
                # Handle NaN/inf values
                if np.isnan(val) or np.isinf(val):
                    values.append(0)
                else:
                    values.append(val)
            
            label = _label_from_key(agent)
            
            # Plot bars
            bars = ax.bar(x + i * width, values, width, 
                          label=label, color=colors.get(agent, '#333333'))
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Phase')
        ax.set_ylabel(kpi)
        ax.set_title(f'{kpi} by Disruption Phase')
        ax.set_xticks(x + width * (len(agents) - 1) / 2)
        ax.set_xticklabels(['Pre-disruption', 'During disruption', 'Post-disruption'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Phase-wise Analysis of Disruption Impact (RQ2)', fontsize=14)
    plt.tight_layout()
    
    # Save
    outdir = os.path.join(config.model_dir, 'saved_figures')
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, 'phase_analysis.pdf')
    plt.savefig(fname, format='pdf', dpi=300)
    print(f"Phase analysis plot saved: {fname}")
    plt.close()

def create_all_plots(final_results, config):
    """
    Create all plots at the end of training.
    """
    # Plot 1: Training progress
    if 'training_history' in final_results and final_results['training_history']:
        plot_training_progress(final_results['training_history'], config)
    
    # Plot 2: KPI comparison
    if 'kpi_summary' in final_results:
        plot_kpi_comparison(final_results['kpi_summary'], config)
    
    # Plot 3: Inventory vs demand
    if 'test_histories' in final_results and 'disruption_info' in final_results:
        plot_inventory_vs_demand(final_results['test_histories'], 
                               final_results['disruption_info'], 
                               config)
    
    # Plot 4: Phase analysis
    if 'phase_kpis' in final_results:
        plot_phase_analysis(final_results['phase_kpis'], config)
    
    print(f"\nAll plots saved to: {os.path.join(config.model_dir, 'saved_figures')}")