# Early Warning-Aware Deep Q-Network for Supply Chain Disruption Management

This project adapts the Deep Q-Network implementation for the Beer Game by Oroojlooyjadid et al. (2021) to develop an Early Warning-Aware DQN (EWA-DQN) that learns to respond to supply chain disruption warnings.

## Installation and Usage

### Prerequisites
- Python 3.7 or higher
- Conda (recommended) or pip
- TensorFlow 1.15 (Note: This code does not work with TensorFlow 2+)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone [repository-url]
   cd [repository-name]
   ```

2. **Create a new conda environment**
   ```bash
   conda create -n ewa python=3.7
   conda activate ewa
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Experiments

1. **Train the models**
   ```bash
   python main.py
   ```
   This will:
   - Evaluate the Sterman heuristic baseline
   - Train a standard DQN agent
   - Train three EWA-DQN variants with different warning accuracies (90%, 80%, 70%)
   - Save all results in `logs/run_YYYYMMDD_HHMMSS/`
   - Display training progress in the console

2. **Analyse the results**
   ```bash
   python data_analyser.py
   ```
   This will:
   - Generate performance comparison plots
   - Create confidence interval visualisations
   - Produce summary statistics
   - Save all figures in `logs/run_YYYYMMDD_HHMMSS/combined_analysis/`

   To analyse a specific run:
   ```bash
   python data_analyser.py --log_dir logs/run_YYYYMMDD_HHMMSS
   ```

### Expected Output
- Training progress will be displayed in the console with periodic test results
- A timestamped directory under `logs/` containing:
  - Model checkpoints
  - Training logs in JSON/CSV format
  - Performance plots (PDF format)
  - Summary statistics

## Key Adaptations from Original Code

This implementation significantly modifies the original Beer Game RL code to focus on early warning systems for supply chain disruptions.

### New Files Added

1. **training_logger.py**
   - Comprehensive logging system for training data
   - Saves training episodes, test results, and phase analysis
   - Exports to JSON and CSV formats
   - Includes confidence interval tracking
   - Creates summary reports

2. **early_stopping_utils.py**
   - PlateauDetector class for detecting performance plateaus
   - Implements coefficient of variation-based stopping criteria
   - Tracks performance statistics over sliding windows

3. **data_analyser.py**
   - Combined analysis tool for post-training visualization
   - Creates multiple plots: training progress, EWA sensitivity, performance comparison
   - Generates confidence interval visualisations
   - Produces summary tables and reports

4. **metrics.py**
   - Implements new KPI calculations: Bullwhip Ratio, Ripple Effect, Bullwhip-Ripple Index
   - Phase-based analysis functions (pre/during/post disruption)
   - Safe variance calculations to handle edge cases

5. **env/disruption.py**
   - DisruptionProcess class implementing supply chain disruptions
   - 50% capacity reduction during disruption windows
   - Timing constraints to ensure learning opportunities

6. **env/warning.py**
   - EarlyWarningSignal class for disruption warnings
   - Probabilistic warning generation with pd (detection) and pf (false positive) rates
   - Warning window starts 2 periods before actual disruption

### Modified Files

7. **config.py** (SIGNIFICANTLY MODIFIED)
   - Simplified to single-agent setup (removed multi-agent parameters)
   - Added EWA-specific parameters: `pd`, `pf`, `stateDimEWA`
   - Added early stopping parameters: `patience`, `min_episodes`
   - Modified hyperparameters for better convergence
   - Added EWA-specific learning rates and warmup periods
   - Removed unnecessary complexity from original multi-agent setup

8. **main.py** (COMPLETELY REWRITTEN)
   - Implemented 3-phase training: Sterman baseline → Standard DQN → EWA variants
   - Added performance tracking and adaptive training
   - Implemented collapse detection and recovery mechanisms
   - Added plateau-based early stopping
   - Integrated comprehensive logging and visualisation
   - Changed from multi-agent to single-agent focus
   - Added confidence interval evaluation

9. **clBeergame.py** (SIGNIFICANTLY MODIFIED)
   - Simplified to single-agent (retailer only) setup
   - Integrated DisruptionProcess and EarlyWarningSignal
   - Added KPI calculation during evaluation
   - Modified game flow to incorporate disruptions
   - Added phase-based analysis support
   - Removed multi-agent coordination logic
   - Enhanced test evaluation with statistical analysis

10. **BGAgent.py** (SIGNIFICANTLY MODIFIED)
    - Simplified to single agent logic
    - Added warning signal integration
    - Completely redesigned reward function with:
      - Anti-collapse mechanisms
      - EWA-specific reward shaping
      - Warning-aware bonuses/penalties
    - Added performance tracking
    - Modified state representation for standard vs EWA agents
    - Added warning history tracking

11. **SRDQN.py** (SUBSTANTIALLY MODIFIED)
    - Added PrioritizedReplayBuffer for better learning
    - Implemented EWADQN as a separate class
    - Added anti-collapse mechanisms
    - Implemented adaptive epsilon decay
    - Added performance-based model saving
    - Modified architecture for EWA (larger hidden layers)
    - Fixed state dimension handling
    - Added emergency override mechanisms

12. **plotting.py** (MODIFIED)
    - Dynamic handling of multiple EWA variants
    - Enhanced visualisation for phase analysis
    - Added inventory vs demand plots
    - Improved KPI comparison charts
    - Better colour mapping for consistency

13. **utilities.py** (SIMPLIFIED)
    - Removed complex multi-agent directory structure
    - Simplified logging setup
    - Enhanced config saving with better error handling

## License and Attribution

This project is an adaptation of the Deep Q-Network implementation for the Beer Game by Oroojlooyjadid et al. (2021).

### Original Work
- **Original Repository**: [Beer Game RL](https://github.com/OptMLGroup/DeepBeerInventory-RL)
- **Original Authors**: Afshin Oroojlooyjadid, MohammadReza Nazari, Lawrence V. Snyder, Martin Takáč
- **Original Paper**: [A Deep Q-Network for the Beer Game: Deep Reinforcement Learning for Inventory Optimization](https://doi.org/10.1287/msom.2020.0939)
- **Original License**: BSD 3-Clause License
- **Copyright**: (c) 2020, Optimization and Machine Learning Group @ Lehigh

### Citation
```bibtex
@article{oroojlooyjadid2021deep,
  title={A Deep Q-Network for the Beer Game: Deep Reinforcement Learning for Inventory Optimization},
  author={Oroojlooyjadid, Afshin and Nazari, MohammadReza and Snyder, Lawrence V. and Tak{\'a}{\v{c}}, Martin},
  journal={Manufacturing \& Service Operations Management},
  volume={24},
  number={1},
  pages={285--304},
  year={2022},
  publisher={INFORMS},
  doi={10.1287/msom.2020.0939}
}
```

### License Notice
This adapted work maintains the BSD 3-Clause License from the original repository. The full license text is available in the LICENSE file.

Copyright (c) 2020, Optimization and Machine Learning Group @ Lehigh (for original components)  
All rights reserved.

