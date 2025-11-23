import torch
import random
import numpy as np
from collections import deque
from environment import Action, Coords, BLOCK_SIZE
from neural_net import DeepQNetwork
import torch.optim as optim
import torch.nn as nn

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class BaseAgent:
    def __init__(self):
        self.n_games = 0

    def get_state(self, game):
        head = game.snake[0]
        
        # points around head
        pt_l = Coords(head.x - BLOCK_SIZE, head.y)
        pt_r = Coords(head.x + BLOCK_SIZE, head.y)
        pt_u = Coords(head.x, head.y - BLOCK_SIZE)
        pt_d = Coords(head.x, head.y + BLOCK_SIZE)
        
        # curr direction 
        dir_l = game.direction == Action.LEFT
        dir_r = game.direction == Action.RIGHT
        dir_u = game.direction == Action.UP
        dir_d = game.direction == Action.DOWN

        state = [
            # danger straight
            (dir_r and game.check_collision(pt_r)) or 
            (dir_l and game.check_collision(pt_l)) or 
            (dir_u and game.check_collision(pt_u)) or 
            (dir_d and game.check_collision(pt_d)),

            # danger right
            (dir_u and game.check_collision(pt_r)) or 
            (dir_d and game.check_collision(pt_l)) or 
            (dir_l and game.check_collision(pt_u)) or 
            (dir_r and game.check_collision(pt_d)),

            # danger left
            (dir_d and game.check_collision(pt_r)) or 
            (dir_u and game.check_collision(pt_l)) or 
            (dir_r and game.check_collision(pt_u)) or 
            (dir_l and game.check_collision(pt_d)),
            
            # Move Direction
            dir_l, dir_r, dir_u, dir_d,
            
            # Location relative to Head
            game.food.x < game.head.x, # Food Left
            game.food.x > game.head.x, # Food Right
            game.food.y < game.head.y, # Food Up
            game.food.y > game.head.y  # Food Down
        ]
        return np.array(state, dtype=int)

class RandomAgent(BaseAgent):
    def select_action(self, state):
        action = [0, 0, 0]
        action[random.randint(0, 2)] = 1
        return action
    
    def train_short(self, *args): pass
    def train_long(self, *args): pass

class SarsaAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.epsilon = 0
        self.gamma = 0.9
        self.lr = 0.1
        self.q_table = {}

    def _get_key(self, state):
        return tuple(state)

    # simple epsilon decay based on game count
    def select_action(self, state):
        self.epsilon = 80 - self.n_games
        if random.randint(0, 200) < self.epsilon:
            idx = random.randint(0, 2)
        else:
            key = self._get_key(state)
            if key not in self.q_table:
                self.q_table[key] = [0.0, 0.0, 0.0]
            idx = np.argmax(self.q_table[key])
        
        action = [0, 0, 0]
        action[idx] = 1
        return action

    def train_step(self, state, action, reward, next_state, next_action, done):
        key = self._get_key(state)
        next_key = self._get_key(next_state)
        
        idx = np.argmax(action)
        next_idx = np.argmax(next_action)

        if key not in self.q_table: self.q_table[key] = [0.0] * 3
        if next_key not in self.q_table: self.q_table[next_key] = [0.0] * 3

        q_current = self.q_table[key][idx]
        q_next = self.q_table[next_key][next_idx]
        
        target = reward if done else (reward + self.gamma * q_next)
        self.q_table[key][idx] += self.lr * (target - q_current)

class DQNAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.net = DeepQNetwork(11, 512, 3) # simple dqn architecture
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        pred = self.net(state)
        target = pred.clone()

        # update target values
        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:
                Q_new = reward[i] + self.gamma * torch.max(self.net(next_state[i]))
            target[i][torch.argmax(action[i]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

    def train_long(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        s, a, r, ns, d = zip(*mini_sample)
        self.train_step(s, a, r, ns, d)

    def train_short(self, state, action, reward, next_state, done):
        self.train_step(state, action, reward, next_state, done)

    def select_action(self, state):
        # epsilon-greedy action selection
        self.epsilon = 80 - self.n_games
        action = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            idx = random.randint(0, 2)
            action[idx] = 1
        else:
            state_t = torch.tensor(state, dtype=torch.float)
            prediction = self.net(state_t)
            idx = torch.argmax(prediction).item()
            action[idx] = 1
        return action