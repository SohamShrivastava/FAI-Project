# 🐍 Reinforcement Learning Snake Game

A complete reinforcement-learning project implementing Random, SARSA, and Deep Q-Network (DQN) agents to play the classic Snake game.
Built using Python, PyTorch, and Pygame.

---

## 🚀 Overview

This repository contains a modular RL framework for training different agents on a custom Snake environment.

### ✔ Features

- Custom Snake environment using **Pygame**
- Three agent types: **Random**, **SARSA**, **DQN**
- Replay memory + batch training for DQN
- Neural network with checkpoint saving
- Clean, extensible code structure

---

## 📂 Project Structure

├── agents.py         
├── environment.py   
├── neural_net.py     
├── run_experiment.py           
└── README.md

---

## 🎮 Snake Environment (`environment.py`)

### 📌 **State Representation (11 Features)**

Each observation is encoded as an **11-dimensional binary feature vector**:

#### 🛑 Danger Awareness
1. Danger straight  
2. Danger right  
3. Danger left  

#### 🧭 Current Movement Direction
4. Moving left  
5. Moving right  
6. Moving up  
7. Moving down  

#### 🍎 Food Position (Relative to Head)
8. Food left  
9. Food right  
10. Food up  
11. Food down  

---

### 💀 **Game Termination Conditions**

An episode ends when:

- 🧱 Snake hits the wall  
- 🌀 Snake collides with its own body  
- ⏳ Too many steps without eating (starvation)


---

## 🧠 Reinforcement Learning Agents (`agents.py`)

### 🎲 **RandomAgent**
A simple baseline agent that:
- Chooses actions uniformly at random  
- Provides a reference for measuring RL improvements  

---

### 📘 **SARSA Agent – Tabular SARSA(0)**

Implements **on-policy Temporal Difference learning**.

#### Key Features:
- ε-greedy action selection (ε decays over time)
- Q-table stored as:  
  **state → [Q(a₀), Q(a₁), Q(a₂)]**
- Online update rule after each step

---

### 🤖 **DQN Agent – Deep Q-Learning**

A neural network–based agent capable of learning complex strategies.

#### 🔧 Architecture:
- Input → Hidden (512 ReLU) → Output  
  **11 → 512 → 3**
