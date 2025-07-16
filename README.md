# SELA: Smart Edge LLM Agent to Optimize Response Trade-offs of AI Assistants

The advent of Large Language Models (LLMs) has changed the way we process information today and has unlocked new ways of delivering intelligence to the user. One of the ways of interfacing with AI is via smart assistants and chatbots that take multimodal inputs. However, the diversity of input tasks imply the possibility of both latency-critical and complex input instructions for AI assistants. Further, LLMs cannot be deployed on the edge for low-latency outputs, as that presents challenges due to their high computational demands and memory requirements. This work explores such trade-offs and contributes a smart LLM selection policy, called SELA, that leverages a suite of LLM models with disparate characteristics to optimize overall quality of service (QoS). SELA uses a time-criticality and complexity predictor at the edge to identify the optimal LLM choice for a given input instruction. Experiments on public instruction benchmarks demonstrate that SELA provides 9% to 62% higher QoS scores compared to the state-of-the-art selection policies. 

# Quick Start Guide

## Installation

```bash
pip3 install -r requirements.txt
```

## Model Training

```bash
python3 train_llm.py --model FCNNet
python3 train_llm.py --model EENet
python3 train_llm.py --model BranchyNet
python3 train_llm.py --model ZTWNet
```

## Selection Training

```bash
python3 train_rl.py --model DQNPolicy
python3 train_rl.py --model PPOPolicy
python3 train_rl.py --model SACPolicy
python3 train_rl.py --model DDPGPolicy
```

## Test

```bash
python3 test.py
```

## Cite this work

Our paper is available in the Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies.
If you use this work, please cite using the following bibtex entry.
```bibtex
@article{tuli2025sela,
  title={{SELA: Smart Edge LLM Agent to Optimize Response Trade-offs of AI Assistants}},
  author={Tuli, Shreshth and Roveri, Manuel and Casale, Giuliano},
  journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume={9},
  number={3},
  year={2025}
}
```

## License

BSD-3-Clause. 
Copyright (c) 2025, Shreshth Tuli.
All rights reserved.
