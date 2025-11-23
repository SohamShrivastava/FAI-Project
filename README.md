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
📌 State Representation (11 Features)

Each state is encoded as an 11-dimensional binary vector, providing compact but rich information:

Category	                            Features
Danger Awareness	                    1. Danger straight
                                      2. Danger right
                                      3. Danger left
                                      
Current Movement Direction	          4. Moving left
                                      5. Moving right
                                      6. Moving up
                                      7. Moving down
                                      
Food Location (Relative to Head)	    8. Food left
                                      9. Food right
                                      10. Food up
                                      11. Food down


💀 Game Termination Conditions

The episode ends when any of the following occurs:

  🧱 Snake hits the wall
  🌀 Snake collides with its own body
  ⏳ Starvation


🧠 Reinforcement Learning Agents (agent.py)

🎲 RandomAgent

A simple baseline agent that:
  1. Selects actions uniformly at random
  2. Provides a reference point for evaluating learning agents
     

📘 SARSA Agent (Tabular SARSA(0))

Implements a classical on-policy Temporal Difference method.

Key Features:
  1. ε-greedy exploration (decays with number of games)
  2. Tabular Q-learning structure


🤖 DQN Agent (Deep Q-Learning)
A neural network–based agent capable of learning advanced strategies.
