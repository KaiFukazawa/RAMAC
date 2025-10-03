# RAMAC（Risk-Aware Multimodal Actor Critic） for Offline Reinforcement Learning

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)

## Overview

**Title**: RAMAC: Multimodal Risk-Aware Offline Reinforcement Learning and the Role of Behavior Regularization

**Authors**: Kai Fukazawa, Kunal Mundada, and Iman Soltani

**Paper**: [https://www.arxiv.org/abs/2510.02695](https://www.arxiv.org/abs/2510.02695)

This repository contains the official implementation of risk-aware expressive generative policies for offline reinforcement learning. Our work introduces novel algorithms that combine:

- **Risk-aware learning** using digstributional RL techniques (CVaR, Wang, CPW, Power utility)
- **Diffusion and Flow-Matching policy modeling** for improved expressiveness
- **Offline RL** capabilities for learning from static datasets from static datasets, with behavior cloning to regularize out-of-distribution (OOD) actions

## Algorithms Implemented

| Algo | File / Dir | Note |
|------|------------|------|
| **RADAC**  | `agents/radac.py` | Diffusion-Actor + CVaR / Wang / CPW /Power|
| **RAFMAC** | `agents/rafmac.py` | Flow-Matching actor |

All algorithms share a **common runner** (`main.py`) and a simple YAML configuration system, making it easy to extend and modify for additional experiments.

## Installation

### Step 1: Clone Repository
```bash
git clone git@github.com:KaiFukazawa/RAMAC.git
cd RAMAC
```

### Step 2: Create Environment

We have used *conda* to create the environment, and would recommend using it as well for better reproducibility.

```bash
# Option 1: Using conda (recommended)
conda create -n radac-env python=3.9.0
conda activate radac-env
```

### Step 3: Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install MuJoCo (automated script)
chmod +x install_mujoco.sh
./install_mujoco.sh

# Add the path to MuJoCo
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
```

### Step 4: Verify Installation
```bash
# Quick sanity check (should complete quickly relative to full training)
python main.py --env_name halfcheetah-medium-expert-v2 --algo radac --device 0 --seed 0 --max_timesteps 100000
```

## Quick Start Guide

### Basic Training
```bash
# Train RADAC on HalfCheetah medium-expert dataset
python main.py --env_name halfcheetah-medium-expert-v2 --algo radac --device 0 --seed 0 --exp exp_1

# Train RAFMAC (flow-matching variant)
python main.py --env_name halfcheetah-medium-replay-v2 --algo rafmac --device 0 --seed 0 --save_best_model --exp exp_2
```

### Working with Risky Environments
```bash
# Step 1: Create risky dataset wrapper
python -m environment.risky_create_hdf5 --config environment/configs/halfcheetah-medium-expert-v2.json

# Step 2: Train with risky evaluation
python main.py \
    --env_name halfcheetah-medium-expert-v2 \
    --risky_dataset_path risky_d4rl/halfcheetah_medium_expert.hdf5 \
    --eval_risky_env \
    --algo radac \
    --device 0 \
    --seed 0 \
    --save_best_model \
    --save_model \
    --exp experiment_name
```

<!-- ### Evaluation
```bash
# Evaluate trained models across multiple seeds
python evaluate_multiple_seeds.py \
    --env_name halfcheetah-medium-replay-v2 \
    --algo radac \
    --model_root saved_models/halfcheetah-medium-replay-v2_experiment_name \
    --num_seeds 5 \
    --episodes_per_seed 10
``` -->

<!-- ## 📊 Reproducing Paper Results

### Main Results Tables
```bash
# Calculate metrics for all algorithms on both environments
python reproduce_paper_results.py --config discover --table_type env
```

### Evaluation details

The reproduction script evaluates each trained model (algo + environment) across 50 independent rollouts (10 episodes per seed, 5 seeds total), providing robust statistical assessment and controlling for initialization-dependent variance in policy performance.

However, there are some differences in the number of steps taken by the models in the evaluation episodes. In our evaluation script in main.py, we use 1000-step limit per episode, however script in the `reproduce_paper_results.py` uses an environment-specific step limit: 200 steps for HalfCheetah, 500 steps for Walker2d/Hopper environments. -->

## Usage Guidelines

### Configuration System
This project uses a simple YAML-based configuration system for managing hyperparameters:

- **Main config file:** `configs/rafmac.yaml` & `configs/radac.yaml` - Contains algorithm-specific hyperparameters
- **Usage:** Parameters are automatically loaded based on the environment name (e.g., `--env_name halfcheetah-medium-v2`)
- **Structure:** Each environment has its own section with specific learning rates, network settings, and risk parameters

You can override config file location using: `--config path/to/your/config_directory`

### Adding New Algorithms
 To add a new algorithm, follow these steps:

1. Create a new YAML file in the `configs` directory with the algorithm name.
2. Define the hyperparameters for the new algorithm in the YAML file.
4. Implement the new algorithm in the `agents` directory.

## References
This project builds upon several excellent open-source projects:
- [D4RL](https://github.com/rail-berkeley/d4rl) for offline RL datasets
- [MuJoCo](https://github.com/deepmind/mujoco) for physics simulation
- [Diffusion-QL](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL)
- [ORAAC](https://github.com/nuria95/O-RAAC)

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{fukazawa2025ramacmultimodalriskawareoffline,
      title={RAMAC: Multimodal Risk-Aware Offline Reinforcement Learning and the Role of Behavior Regularization}, 
      author={Kai Fukazawa and Kunal Mundada and Iman Soltani},
      year={2025},
      eprint={2510.02695},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.02695}, 
}
```
---
