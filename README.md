# 🚀 Iterative AI Projects: Pathfinding, Probabilistic Sensing, and Herding

This repository contains three Artificial Intelligence projects. Each part focuses on a different AI strategy and application in a dynamic environment with agents, obstacles, and objectives.

## 🧠 Part 1: A* Based Bot Navigation

### Problem
Navigate a bot to rescue the captain in a ship grid while avoiding dynamically moving aliens.

### Key Bots
- **Bot1**: Plans the shortest path at the beginning using A*, ignoring alien movements.
- **Bot2**: Replans the path at every timestep based on the updated alien positions.
- **Bot3**: Avoids alien-adjacent cells if possible while planning.
- **Bot4**: Dynamically adjusts paths using a custom heuristic to maximize safety and success.

### Highlights
- Grid generation with blocked and open cells.
- Visualization using color-coded maps.
- Analysis of survival and success rate vs. number of aliens.

### Performance
Bot4 outperformed other bots in terms of both survival and success rate in high-alien-density environments.

---

## 🧮 Part 2: Bayesian Reasoning for Search & Rescue

### Problem
The bot must rescue crew members using limited, probabilistic sensor information while avoiding aliens.

### Scenarios
- 1 alien, 1 crew
- 1 alien, 2 crew
- 2 aliens, 2 crew

### Bots
- **Bot1**: Uses sensor input and Bayes' theorem to update beliefs and plan path.
- **Bot2**: Custom bot that integrates crew density and alien probability into A* heuristic.
- **Bot3–5**: Handle multi-crew/alien scenarios using 4D belief matrices.

### Core Methods
- Bayesian updates using conditional probabilities.
- Probabilistic modeling of crew beeps and alien sensor data.
- Real-time knowledge base updates.

### Results
- BOT2 showed a significant improvement in average steps and success rates over BOT1.
- Multi-crew bots handled uncertainty with 4D belief networks, improving robustness.

---

## 🔄 Part 3: MDP + Reinforcement Learning for Herding

### Problem
Guide a panicked crew member to a teleport pad using a security bot, minimizing escape time.

### Bot Behavior
- **No Bot**: Crew wanders randomly.
- **Optimal Bot**: Uses value iteration and Bellman updates.
- **Learned Bot**: Approximates the policy learned from value iteration using a model.

### Techniques
- Grid world modeled as an MDP.
- Value iteration to compute expected rescue times (T_bot).
- Optimal policy computed via expected transition rewards.
- Learned agent trained to generalize across configurations.

### Outcome
- Optimal bots drastically reduce expected escape time.
- Learned bots generalize well, offering a memory-efficient alternative.
- Visual comparisons between `T_no_bot`, `T_bot`, and policy matrices.

---

## 📊 Key Takeaways

| Approach      | Strength                          | Weakness                            |
|---------------|-----------------------------------|-------------------------------------|
| A\* Search     | Deterministic, Efficient          | Poor adaptability to dynamic threats |
| Bayesian      | Probabilistic reasoning, flexible | Sensitive to poor sensor models     |
| MDP           | Global optimal policy             | High computation, requires full model |


