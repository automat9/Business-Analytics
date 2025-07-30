#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import random
import numpy as np
import tensorflow as tf
import os
import sys
from collections import defaultdict, deque

# Add custom modules
from config import get_config
from clBeergame import clBeerGame
from utilities import prepare_dirs_and_logger, save_config
from plotting import create_all_plots
from training_logger import TrainingLogger
from early_stopping_utils import PlateauDetector

def gen_demand_sequence(config):
    """Generate demand sequence with controlled variability."""
    T = config.Ttest
    seq = np.empty(T, dtype=int)
    # First 4 periods: stable demand
    seq[:4] = 4
    # Remaining: normal distribution
    noise = np.random.normal(config.demandMu, config.demandSigma, T-4)
    noise = np.round(noise)
    noise = np.clip(noise, 1, 8)
    seq[4:] = noise.astype(int)
    return seq

def evaluate_sterman_baseline(config, test_demands):
    """Evaluate corrected Sterman heuristic as baseline."""
    print("\n" + "="*60)
    print("=== Phase 1: Evaluating CORRECTED Sterman Baseline ===")
    print("="*60)
    
    config.agentTypes = ['Strm']
    beerGame = clBeerGame(config)
    
    print("Sterman formula: Order = AS[t] + α(target_inv - IL) + β(target_pipeline - OO)")
    print(f"With α={config.alpha_b}, β={config.betta_b}, target_inv={4}, target_pipeline={8}")
    
    test_results = beerGame.doTestMid(test_demands, return_results=True)
    
    print(f"\nSterman Baseline Results:")
    print(f"  Cost: {test_results['avg_kpi']['Cost']:.0f}")
    print(f"  BW:   {test_results['avg_kpi']['BW']:.2f}")
    
    return test_results, beerGame.disruption.get_info()

def train_standard_dqn(config, test_demands, logger):
    """Train standard DQN with collapse detection and recovery."""
    print("\n" + "="*60)
    print("=== Phase 2: Training Standard DQN ===")
    print("="*60)
    
    config.agentTypes = ['srdqn']
    config.ifUsePreviousModel = False
    config.iftl = False
    
    beerGame = clBeerGame(config)
    
    # Performance tracking
    best_cost = float('inf')
    best_model_path = None
    episodes_without_improvement = 0
    performance_history = deque(maxlen=20)
    collapse_count = 0
    plateau_detector = PlateauDetector(window_size=10, variance_threshold=0.05, min_episodes=2000)
    
    print(f"Training configuration:")
    print(f"  Learning rate: {config.lr0}")
    print(f"  Initial epsilon: {config.epsilonBeg}")
    print(f"  Batch size: {config.batchSize}")
    print(f"  Target update: every {config.dnnUpCnt} steps")
    print(f"  Collapse detection threshold: 10,000")
    
    for ep in range(config.maxEpisodesTrain):
        # Generate training demand
        train_demand = gen_demand_sequence(config)
        reward = beerGame.playGame(train_demand, "train")
        episode_cost = -reward * 100  # Unnormalize for tracking
        
        # Update DQN's performance tracker
        if hasattr(beerGame.player, 'brain'):
            beerGame.player.brain.update_performance(episode_cost)
        
        # Track performance
        performance_history.append(episode_cost)
        
        # Log episode
        logger.log_episode('srdqn', ep + 1, reward, train_demand,
                         beerGame.disruption.get_info())
        
        # Periodic testing and evaluation
        if (ep + 1) % config.testInterval == 0:
            print(f"\n{'='*50}")
            print(f"Episode {ep + 1}/{config.maxEpisodesTrain}")
            
            # Performance diagnostics
            recent_avg = np.mean(performance_history) if performance_history else 0
            print(f"Recent training avg cost: {recent_avg:.0f}")
            
            # Check for collapse
            if recent_avg > 10000:
                collapse_count += 1
                print(f"⚠️  PERFORMANCE COLLAPSE DETECTED (count: {collapse_count})")
                
                # Try to recover from best saved model
                if best_model_path and collapse_count <= 3:
                    try:
                        print(f"🔄 Attempting recovery from best model...")
                        beerGame.player.brain.saver.restore(
                            beerGame.player.brain.session, 
                            best_model_path
                        )
                        # Reset exploration
                        beerGame.player.brain.epsilon = min(0.5, config.epsilonBeg)
                        print(f"✅ Recovered! Reset epsilon to {beerGame.player.brain.epsilon:.2f}")
                    except:
                        print(f"❌ Recovery failed")
            
            # Full test evaluation
            test_results = beerGame.doTestMid(test_demands, return_results=True)
            avg_kpi = test_results['avg_kpi']
            
            # Store progress
            logger.log_test_results('srdqn', ep + 1, test_results,
                                  beerGame.disruption.get_info())
            
            # Update plateau detector
            plateau_detector.update(avg_kpi['Cost'])
            
            # Track best performance
            if avg_kpi['Cost'] < best_cost:
                improvement = best_cost - avg_kpi['Cost']
                best_cost = avg_kpi['Cost']
                episodes_without_improvement = 0
                print(f"✅ New best! Cost: {best_cost:.0f} (improved by {improvement:.0f})")
                
                # Save best model
                if hasattr(beerGame.player, 'brain'):
                    best_model_path = os.path.join(
                        beerGame.player.brain.address, 
                        f'best-model-cost{int(best_cost)}'
                    )
                    # Save the path that saver.save() returns
                    saved_path = beerGame.player.brain.saver.save(
                        beerGame.player.brain.session,
                        best_model_path
                    )
                    best_model_path = saved_path  # Use the actual saved path
                    
                # Reset collapse count on improvement
                if best_cost < 5000:
                    collapse_count = 0
            else:
                episodes_without_improvement += config.testInterval
            
            # Check for plateau
            if plateau_detector.has_plateaued():
                stats = plateau_detector.get_stats()
                print(f"\n📊 Performance plateau detected!")
                print(f"   Last 10 tests: mean={stats['mean']:.0f}, std={stats['std']:.0f}, CV={stats['cv']:.3f}")
                if stats['mean'] < 600:  # Good enough performance
                    print(f"   ✅ Converged at good performance level")
                    break
            
            # Performance feedback
            if avg_kpi['Cost'] < 600:  # Better than Sterman, this is based on numerous test runs
                print(f"🏆 EXCELLENT! Outperforming baseline")
            elif avg_kpi['Cost'] < 1000:
                print(f"👍 Good performance")
            elif avg_kpi['Cost'] > 10000:
                print(f"❌ Poor performance - may need intervention")
            
            # Early stopping with minimum episodes
            if ep + 1 >= config.min_episodes:
                if episodes_without_improvement >= config.patience:
                    print(f"\n⏹️  Early stopping at episode {ep+1}")
                    break
    
    # Final evaluation
    print("\n📊 Final evaluation of standard DQN:")
    final_results = beerGame.doTestMid(test_demands, return_results=True)
    
    # Restore best model for final eval if needed
    if best_model_path and final_results['avg_kpi']['Cost'] > best_cost * 1.5:
        print(f"Restoring best model for final evaluation...")
        beerGame.player.brain.saver.restore(
            beerGame.player.brain.session, 
            best_model_path
        )
        final_results = beerGame.doTestMid(test_demands, return_results=True)
    
    return final_results, best_cost

def train_ewa_dqn(config, test_demands, logger, pd, pf, pretrained_path=None):
    """Train EWA-DQN with specific warning parameters."""
    print("\n" + "="*60)
    print(f"=== Training EWA-DQN (pd={pd}, pf={pf}) ===")
    print("="*60)
    
    # Configure for EWA
    config.pd = pd
    config.pf = pf
    config.agentTypes = ['ewa']
    
    # Create unique model directory
    ewa_key = f'ewa_pd{int(pd*100)}_pf{int(pf*100)}'
    original_model_dir = config.model_dir
    config.model_dir = os.path.join(original_model_dir, ewa_key)
    os.makedirs(config.model_dir, exist_ok=True)
    
    # Adjust hyperparameters for EWA
    if pretrained_path and os.path.exists(pretrained_path):
        config.ifUsePreviousModel = True
        config.lr0 = config.ewa_lr0 * 0.5  # Even lower LR for fine-tuning
        print(f"📦 Loading pretrained weights from DQN...")
    else:
        config.ifUsePreviousModel = False
        config.lr0 = config.ewa_lr0  # EWA-specific learning rate
        print(f"🆕 Training from scratch")
    
    # Use longer warmup for EWA
    config.minReplayMem = config.ewa_warmup
    
    beerGame = clBeerGame(config)
    
    # Training parameters
    best_cost = float('inf')
    best_model_path = None
    episodes_without_improvement = 0
    performance_history = deque(maxlen=20)
    plateau_detector = PlateauDetector(window_size=15, variance_threshold=0.05, min_episodes=5000)
    
    accuracy = (pd - pf) * 100
    print(f"Warning signal accuracy: {accuracy:.0f}%")
    print(f"Detection rate: {pd:.1f}, False alarm rate: {pf:.2f}")
    print(f"Learning rate: {config.lr0}, Warmup: {config.minReplayMem}")
    print(f"State dimension for EWA: 7 (base 6 + warning flag + time since warning)")
    
    for ep in range(config.maxEpisodesTrain):
        train_demand = gen_demand_sequence(config)
        reward = beerGame.playGame(train_demand, "train")
        episode_cost = -reward * 100
        
        performance_history.append(episode_cost)
        
        # Update performance tracker
        if hasattr(beerGame.player, 'brain'):
            beerGame.player.brain.update_performance(episode_cost)
        
        # Log episode
        logger.log_episode(ewa_key, ep + 1, reward, train_demand,
                         beerGame.disruption.get_info())
        
        # Periodic testing
        if (ep + 1) % config.testInterval == 0:
            print(f"\n📊 Episode {ep + 1}")
            
            recent_avg = np.mean(performance_history)
            print(f"Recent training avg: {recent_avg:.0f}")
            
            # Test evaluation
            test_results = beerGame.doTestMid(test_demands, return_results=True)
            avg_kpi = test_results['avg_kpi']
            
            # Log results
            logger.log_test_results(ewa_key, ep + 1, test_results,
                                  beerGame.disruption.get_info())
            
            # Update plateau detector
            plateau_detector.update(avg_kpi['Cost'])
            
            # Track improvement
            if avg_kpi['Cost'] < best_cost:
                improvement = best_cost - avg_kpi['Cost']
                best_cost = avg_kpi['Cost']
                episodes_without_improvement = 0
                print(f"✅ New best! Cost: {best_cost:.0f} (improved by {improvement:.0f})")
                
                # Save best model
                if hasattr(beerGame.player, 'brain'):
                    best_model_path = os.path.join(
                        beerGame.player.brain.address, 
                        f'best-ewa-{int(best_cost)}'
                    )
                    # Save the path that saver.save() returns
                    saved_path = beerGame.player.brain.saver.save(
                        beerGame.player.brain.session,
                        best_model_path
                    )
                    best_model_path = saved_path  # Use the actual saved path
            else:
                episodes_without_improvement += config.testInterval
            
            # Check for plateau with good performance
            if plateau_detector.has_plateaued():
                stats = plateau_detector.get_stats()
                print(f"\n📊 Performance plateau detected!")
                print(f"   Last 15 tests: mean={stats['mean']:.0f}, std={stats['std']:.0f}, CV={stats['cv']:.3f}")
                if stats['mean'] < 600:  # Good enough performance
                    print(f"   ✅ Converged at good performance level")
                    break
                elif ep + 1 >= 8000:  # Don't train forever
                    print(f"   ⏹️ Stopping after sufficient training")
                    break
            
            # Performance feedback
            if avg_kpi['Cost'] < 600:  # Better than Sterman
                print(f"🏆 EXCELLENT! Outperforming baseline")
            elif avg_kpi['Cost'] < 1000:
                print(f"👍 Good performance")
            elif avg_kpi['Cost'] > 10000:
                print(f"❌ Poor performance - may need intervention")
                # Try recovery if we have a good model saved
                if best_model_path and best_cost < 2000:
                    print(f"🔄 Attempting recovery from best model...")
                    try:
                        beerGame.player.brain.saver.restore(
                            beerGame.player.brain.session,
                            best_model_path
                        )
                        # Reset exploration moderately
                        beerGame.player.brain.epsilon = min(0.2, config.epsilonBeg * 0.3)
                        print(f"✅ Recovered! Reset epsilon to {beerGame.player.brain.epsilon:.2f}")
                    except:
                        print(f"❌ Recovery failed")
            
            # Early stopping for EWA with longer patience
            if ep + 1 >= 5000 and episodes_without_improvement >= 3000:
                print(f"\n⏹️  Early stopping at episode {ep+1}")
                break
    
    # Final evaluation
    print(f"\n📊 Final evaluation of EWA-DQN (pd={pd}, pf={pf}):")
    final_results = beerGame.doTestMid(test_demands, return_results=True)
    
    # Restore best model if final evaluation is poor
    if best_model_path and final_results['avg_kpi']['Cost'] > best_cost * 1.5:
        print(f"Restoring best model for final evaluation...")
        import tensorflow as tf
        if tf.train.checkpoint_exists(best_model_path):
            beerGame.player.brain.saver.restore(
                beerGame.player.brain.session,
                best_model_path
            )
            final_results = beerGame.doTestMid(test_demands, return_results=True)
        else:
            print(f"Warning: Could not find checkpoint at {best_model_path}")
            # ⬇️ Fallback to the latest available checkpoint in this model directory
            latest = tf.train.latest_checkpoint(beerGame.player.brain.address)
            if latest:
                beerGame.player.brain.saver.restore(
                    beerGame.player.brain.session,
                    latest
                )
                final_results = beerGame.doTestMid(test_demands, return_results=True)
            else:
                print("Warning: no checkpoints found; using current weights.")

    
    # Restore original config
    config.model_dir = original_model_dir
    
    return final_results, best_cost

def main(config):
    # Set random seeds
    random.seed(config.seed_run)
    np.random.seed(config.seed_run)
    tf.set_random_seed(config.seed_run)

    # Prepare directories and logging
    prepare_dirs_and_logger(config)
    save_config(config)
    
    # Initialize logger
    logger = TrainingLogger(config)
    
    # Pre-generate test demands
    print(f"Generating {config.testRepeatMid} test demand sequences...")
    test_demands = [gen_demand_sequence(config) for _ in range(config.testRepeatMid)]
    
    # Results storage
    final_results = {
        'kpi_summary': {},
        'test_histories': {},
        'training_history': defaultdict(lambda: {'episodes': [], 'costs': [], 'bw': []}),
        'phase_kpis': {},
        'disruption_info': None
    }

    # Phase 1: Corrected Sterman baseline
    sterman_results, disruption_info = evaluate_sterman_baseline(config, test_demands)
    final_results['kpi_summary']['Strm'] = sterman_results['avg_kpi']
    final_results['test_histories']['Strm'] = sterman_results['histories'][0]
    final_results['phase_kpis']['Strm'] = sterman_results['phase_kpis']
    final_results['disruption_info'] = disruption_info
    logger.log_final_evaluation('Strm', sterman_results)
    
    sterman_cost = sterman_results['avg_kpi']['Cost']
    print(f"\n📌 Baseline to beat: Sterman cost = {sterman_cost:.0f}")

    # Phase 2: Train standard DQN
    dqn_results, dqn_best_cost = train_standard_dqn(config, test_demands, logger)
    final_results['kpi_summary']['srdqn'] = dqn_results['avg_kpi']
    final_results['test_histories']['srdqn'] = dqn_results['histories'][0]
    final_results['phase_kpis']['srdqn'] = dqn_results['phase_kpis']
    logger.log_final_evaluation('srdqn', dqn_results)
    
    # Check if DQN succeeded
    dqn_model_path = None
    if dqn_best_cost < sterman_cost:
        print(f"\n✅ DQN succeeded! Best cost {dqn_best_cost:.0f} < Sterman {sterman_cost:.0f}")
        dqn_model_path = os.path.join(config.model_dir, 'model1')
    else:
        print(f"\n⚠️  DQN struggled. Best cost {dqn_best_cost:.0f} >= Sterman {sterman_cost:.0f}")

    # Phase 3: Train EWA-DQN variants
    ewa_scenarios = [
        (0.9, 0.01),   # High accuracy (89%)
        (0.8, 0.05),   # Medium accuracy (75%)
        (0.7, 0.1),    # Low accuracy (60%)
    ]
    
    for pd, pf in ewa_scenarios:
        # Don't use pretrained weights - let EWA learn from scratch
        # This allows it to discover its own warning-aware strategy
        ewa_results, ewa_best = train_ewa_dqn(
            config, test_demands, logger, pd, pf,
            pretrained_path=None  # Changed from using DQN pretrained model
        )
        
        ewa_key = f'ewa_pd{int(pd*100)}_pf{int(pf*100)}'
        final_results['kpi_summary'][ewa_key] = ewa_results['avg_kpi']
        final_results['test_histories'][ewa_key] = ewa_results['histories'][0]
        final_results['phase_kpis'][ewa_key] = ewa_results['phase_kpis']
        logger.log_final_evaluation(ewa_key, ewa_results)

    # Final summary
    print("\n" + "="*60)
    print("=== FINAL SUMMARY ===")
    print("="*60)
    
    print("\nTotal Costs (lower is better):")
    sorted_agents = sorted(final_results['kpi_summary'].items(), 
                          key=lambda x: x[1]['Cost'])
    for agent, kpis in sorted_agents:
        label = agent.replace('_', ' ').replace('pd', 'pd=0.').replace('pf', ', pf=0.')
        print(f"  {label:25s}: {kpis['Cost']:7.0f}")
    
    print("\nBullwhip Ratios (lower is better):")
    for agent, kpis in sorted_agents:
        label = agent.replace('_', ' ').replace('pd', 'pd=0.').replace('pf', ', pf=0.')
        print(f"  {label:25s}: {kpis['BW']:7.2f}")
    
    # Highlight winner
    winner = sorted_agents[0]
    print(f"\n🏆 Best performer: {winner[0]} with cost {winner[1]['Cost']:.0f}")
    
    # Generate plots
    print("\n📊 Generating plots...")
    create_all_plots(final_results, config)
    
    # Save logs
    logger.save_all()
    logger.create_summary_report()
    
    print(f"\n✅ Complete! Results saved in: {config.model_dir}")

if __name__ == '__main__':
    cfg = get_config()
    main(cfg)