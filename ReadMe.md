# Ethical MARL for Disaster Relief

This project implements an ethically-aligned Multi-Agent Reinforcement Learning (MARL) framework for autonomous decision-making in disaster relief scenarios. Inspired by the EU Guidelines for Trustworthy AI, the system trains agents to make rescue decisions that balance fairness, transparency, and reliability—critical ethical dimensions—while navigating high-stakes, uncertain environments.

## Overview

Traditional reinforcement learning agents maximize rewards, often ignoring moral trade-offs. In contrast, this system embeds ethical metrics directly into the reward structure. The framework includes:

- A custom OpenAI Gym environment that simulates disaster scenarios
- A MAPPO-based actor-critic learning setup
- Fairness, transparency, and reliability metrics integrated during training
- Evaluation logs to measure ethical alignment per agent decision

---

## Repository Structure

```bash
.
├── dataset/
│   └── raw/
│       ├── agents.csv         # Static agent locations
│       ├── victims.csv        # Pool of victim profiles
│       └── dataset.py         # (Optional) Dataset generation logic
├── env/
│   └── ethical_env.py         # Custom Gym environment with ethical score computation
├── mappo/
│   ├── buffer.py              # Experience replay buffer
│   ├── network.py             # Actor-critic neural network architecture
│   ├── trainer.py             # MAPPO training loop with PPO loss
│   └── utils.py               # Advantage estimation, seeding, helper functions
├── model/
│   └── logs                   # log files after the model training(include graphs, csv files etc)
├── main.py                    # Entry point for training
├── test.py                    # Evaluation script for testing ethical alignment
```

---

## Ethical Metrics

- **Fairness**: Measures how resources are distributed considering victim severity, urgency, and distance.
- **Transparency**: Derived from the agent’s action confidence (softmax probability).
- **Reliability**: Tracks the consistency of agent decisions across similar ethical contexts.

Agents are rewarded using a weighted combination:

```
Reward = 10 × (0.4 × Fairness + 0.3 × Transparency + 0.3 × Reliability)
```

---

## How to Use

### Prerequisites

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Gym

### Train the Model

```bash
python main.py
```

This will train 40 agents using MAPPO in the custom ethical environment and log training rewards and ethical scores. You can customize the hyperparameters, number of runs (by default 1), number of episodes and steps per episodes in main.py file.

### Run a Test

```bash
python test.py
```

This script:
- Loads a trained actor model
- Samples 3 victims
- Computes fairness, transparency, reliability for each
- Prints whether the agent selected the most ethical victim

---

## Output Example

```text
Model selected victim index: 2
Transparency (confidence): 0.941

Ethical Evaluation:
victim_index | fairness | transparency | reliability | ethical_score | chosen
-------------|----------|--------------|-------------|----------------|-------
     0       | 0.700    |    0.941     |    0.900    |     0.806      | False
     1       | 0.400    |    0.941     |    1.000    |     0.748      | False
     2       | 0.800    |    0.941     |    0.950    |     0.852      | True

Was the choice optimal? Yes
```

---

## Citation

If you use this codebase for academic research, please consider citing:

> Sahil Sandal, "Ethical Decision-Making in Autonomous Multi-Agent Systems for Disaster Relief", Uppsala University Master's Thesis, 2025.

---

## Contact

For questions, reach out via [LinkedIn](https://www.linkedin.com/in/sahil-sandal-46448a138/) or sahilsingh2201@gmail.com.

---

## License

This project is for academic and research purposes. For licensing inquiries, please contact the author.
