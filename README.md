🐍 Reinforcement Learning Snake Game

A complete reinforcement-learning project implementing Random, SARSA, and Deep Q-Network (DQN) agents to play the classic Snake game.
Built using Python, PyTorch, and Pygame.

🚀 Overview

This repository contains a modular RL framework for training different agents on a custom Snake environment.
It includes:

✔ Custom Snake environment with Pygame

✔ Three agent types: Random, SARSA, DQN

✔ Replay memory + batch training for DQN

✔ Neural network model with checkpoint saving

✔ Clear code structure & extensible design

📂 Project Structure
├── agents.py         
├── environment.py   
├── neural_net.py     
├── run_experiment.py           
└── README.md

🎮 Snake Environment (environment.py)
State Representation (size = 11)

Your agent receives an 11-dimensional binary feature vector:

Danger straight

Danger right

Danger left

Moving left

Moving right

Moving up

Moving down

Food left

Food right

Food up

Food down

Action Space (one-hot encoding)
[1, 0, 0] → move straight  
[0, 1, 0] → turn right  
[0, 0, 1] → turn left

Reward Function
Event	Reward
Eating food	+10
Dying (collision or wall)	–10
Normal step	0
Game End Conditions

Snake hits wall

Snake hits its own body

Too many steps without eating (frame_iteration > 100 × length)

🧠 RL Agents (agent.py)
RandomAgent

Baseline model that picks random moves.

SARSA Agent

Tabular SARSA(0):

ε-greedy exploration

Q-table dictionary (state → [Q(a₀), Q(a₁), Q(a₂)])

Online update rule

DQN Agent

Deep Q-learning with:

Replay Memory: 100,000

Batch Size: 1,000

γ = 0.9

Adam optimizer (lr = 0.001)

MSE loss

Two-layer neural network (11 → 512 → 3)
