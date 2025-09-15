# The Hidden Strength of Disagreement: Unraveling the Consensus-Diversity Tradeoff in Adaptive Multi-Agent System

![Workflow](https://github.com/wuzengqing001225/ConsensusDiversityTradeoffMAS/blob/main/IMG/illustration.png?raw=true)

This repository is the source code for our paper **The Hidden Strength of Disagreement: Unraveling the Consensus-Diversity Tradeoff in Adaptive Multi-Agent System**. (EMNLP 2025)

In this work, we investigate how implicit consensus in LLM-based multi-agent systems enables adaptability in dynamic environments by balancing coordination and diversity. Through experiments in *Dynamic Disaster Response*, *Information Spread and Manipulation*, and *Dynamic Public-Goods Provision*, we show that partial deviation from group norms enhances exploration, robustness, and overall performance.

## Setup

**Please do not commit your API key to GitHub, or share it with anyone online.**

To reproduce the results in the paper, please place your API keys in the ```config.json``` files in each case study folder and then run the following command ```python main.py```. You may also change the simulation settings in ```baselines.py``` files of each case study. A mixed model setup is also supported (in disaster response) ```python mixed_model_supported_main.py```.

## Citation

If you find our work useful, please give us credit by citing:

```bibtex
@article{wu2025hidden,
  title={The hidden strength of disagreement: Unraveling the consensus-diversity tradeoff in adaptive multi-agent systems},
  author={Wu, Zengqing and Ito, Takayuki},
  journal={arXiv preprint arXiv:2502.16565},
  year={2025}
}
```
