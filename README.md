# Distributional Crossover for Recurrent Spiking Neural Networks Repository

This repository contains the implementation of the Distributional Crossover evolutional strategies. It includes the Evolutionary Connectivity (EC) algorithm, Recurrent Spiking Neural Networks (RSNN), and the Evolution Strategies (ES) baseline and Distributional Crossover (DC) baseline implemented in JAX.

### Getting Started

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```
or
```bash
conda env create --prefix/-n your_preifx/env_name --file freeze.yml
```

2. [Install JAX](https://github.com/google/jax#installation). **Note**: Jax version of 0.4.19 is proper for DC to run (Whiche are commentted in freeze.yml). You should manually download jax from the official repos. Please check the **CUDA and CUDNN version** when installing jax and jaxlib.

3. [Install W&B](https://github.com/wandb/wandb) and log in to your account to view metrics

### Precautions

- Brax v1 is required (`brax<0.9`) to reproduce our experiments. Brax v2 has completely rewritten the physics engine and adopted a different reward function.
- Due to the inherent numerical stochasticity in Brax's physics simulations, variations in results can occur even when using a fixed seed.

## Usage
    dc-toy-simplified.py contains codes for evaluating in both real RL environment and toy-task environment. 

### Training DC with RSNN
```
scripts/run_dc_toy.sh
```
You can change the params in 'run_dc_toy.sh'. Wandb args like run-names can be changed in the 'dc-toy-simplified.py'. 

### Training EC with RSNN

To set parameters, use the command-line format of [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#id15). For example:

```
python ec.py task=humanoid
```

