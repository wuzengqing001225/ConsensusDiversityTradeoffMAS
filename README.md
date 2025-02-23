# The Hidden Strength of Disagreement: Unraveling the Consensus-Diversity Tradeoff in Adaptive Multi-Agent System

![Workflow](https://github.com/wuzengqing001225/ConsensusDiversityTradeoffMAS/blob/main/IMG/illustration.png?raw=true)

This repository is the source code for our paper **The Hidden Strength of Disagreement: Unraveling the Consensus-Diversity Tradeoff in Adaptive Multi-Agent System**.

In this work, we investigate how implicit consensus in LLM-based multi-agent systems enables adaptability in dynamic environments by balancing coordination and diversity. Through experiments in *Dynamic Disaster Response*, *Information Spread and Manipulation*, and *Dynamic Public-Goods Provision*, we show that partial deviation from group norms enhances exploration, robustness, and overall performance.

## Setup

Please make sure you have the valid OpenAI and Anthropic API key for GPT-4o and Claude-3 models. Otherwise, you may need to apply for one before you test our codes.

**Please do not commit your API key to GitHub, or share it with anyone online.**

To reproduce the results in the paper, please place your API keys in the ```config.json``` files in each case study folder and then run the following command ```python main.py```. You may also change the simulation settings in ```baselines.py``` files of each case study.
