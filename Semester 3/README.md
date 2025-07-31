## License and Attribution

This project is an adaptation of the Deep Q-Network implementation for the Beer Game by Oroojlooyjadid et al. (2021).

### Original Work
- **Original Repository**: [Beer Game RL](https://github.com/OptMLGroup/DeepBeerInventory-RL)
- **Original Authors**: Afshin Oroojlooyjadid, MohammadReza Nazari, Lawrence Snyder, Martin Takáč
- **Original License**: BSD 3-Clause License
- **Copyright**: (c) 2020, Optimization and Machine Learning Group @ Lehigh

### This Adaptation
This work adapts the original codebase to implement an Early Warning-Aware Deep Q-Network (EWA-DQN) for supply chain disruption management. Key modifications include:
- Single-agent focus (retailer only) instead of multi-agent setup
- Integration of supply chain disruption modeling
- Implementation of early warning signals with configurable accuracy
- Enhanced reward engineering for disruption-aware learning
- Comprehensive KPI tracking and statistical analysis
- New visualization and analysis tools

### Citation
If you use this code in your research, please cite both the original work and this adaptation:

**Original Paper:**
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
