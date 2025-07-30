#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
import json
import logging
import numpy as np

def prepare_dirs_and_logger(config):
    """
    Create model directory and set up logging.
    Expects:
      config.model_dir  — base folder for saving models & figures
    """
    # Make sure the model directory exists
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(os.path.join(config.model_dir, 'saved_figures'), exist_ok=True)
    
    # Create subdirectories for each agent's model (we only have agent 0)
    os.makedirs(os.path.join(config.model_dir, 'model1'), exist_ok=True)

    # Configure Python logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

def save_config(config):
    """
    Dump all config parameters to params.json in model_dir.
    Handles non-serializable objects gracefully.
    """
    param_path = os.path.join(config.model_dir, "params.json")
    logging.info(f"Saving config to {param_path}")
    
    # Convert config to dictionary, handling non-serializable objects
    config_dict = {}
    for key, value in vars(config).items():
        try:
            # Test if the value is JSON serializable
            json.dumps(value)
            config_dict[key] = value
        except (TypeError, ValueError):
            # Convert non-serializable objects to string representation
            if isinstance(value, np.ndarray):
                config_dict[key] = value.tolist()
            else:
                config_dict[key] = str(value)
                logging.warning(f"Parameter '{key}' converted to string for JSON serialization")
    
    # Save the config
    try:
        with open(param_path, 'w') as f:
            json.dump(config_dict, f, indent=4, sort_keys=True)
        logging.info(f"Config successfully saved to {param_path}")
    except Exception as e:
        logging.error(f"Failed to save config: {e}")
        # Don't crash the program if config saving fails
        pass