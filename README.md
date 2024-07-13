# Resolving Highway Conflict in Multi-Autonomous Vehicle Controls with Local State Attention

## Algorithms

All the MARL algorithms are extended from the single-agent RL with parameter sharing among agents.
- MAPPO
- IPPO
- MAA2C
- MAPPO_LSA

## Installation
- create an python virtual environment: `conda create -n marl_cav python=3.6 -y`
- active the virtul environment: `conda activate marl_cav`
- install pytorch (torch>=1.2.0): `pip install torch===1.7.0 torchvision===0.8.1 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html`
- install the requirements: `pip install -r requirements.txt`

## Demo
[video demo](https://drive.google.com/file/d/1fW8IFT_w5RL2l4s64auoZi0USMcls0_D/view?usp=sharing)

## Usage
run `main.py`

## Reference
- [Highway-env](https://github.com/eleurent/highway-env)
