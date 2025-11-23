import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

# Use System Font to avoid needing arial.ttf
pygame.init()
FONT = pygame.font.SysFont('arial', 25)

class Action(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Coords = namedtuple('Coords', 'x, y')

# dark theme 
COLOR_BG = (20, 20, 20)
COLOR_SNAKE = (0, 255, 100)
COLOR_FOOD = (255, 50, 50)
COLOR_TEXT = (255, 255, 255)

BLOCK_SIZE = 20
GAME_SPEED = 100  # Faster training

class SnakeEnvironment:
    def __init__(self, width=640, height=480):
        self.w = width
        self.h = height
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('RL Snake Study')
        self.clock = pygame.time.Clock()
        self.reset_env()

    def reset_env(self):
        self.direction = Action.RIGHT
        self.head = Coords(self.w/2, self.h/2)
        
        # small initial snake
        self.snake = [
            self.head,
            Coords(self.head.x-BLOCK_SIZE, self.head.y),
            Coords(self.head.x-(2*BLOCK_SIZE), self.head.y)
        ]

        self.score = 0
        self.food = None
        self._spawn_food()
        self.frame_iteration = 0
        return self.score

    def _spawn_food(self):
        # randomly place food on grid
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        self.food = Coords(x, y)
        if self.food in self.snake:
            self._spawn_food()

    def step(self, action):
        self.frame_iteration += 1
        
        #check for quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        self._update_direction(action)
        self.snake.insert(0, self.head)
        
        reward = 0
        game_over = False

        # Check collision or starvation (iteration limit)
        if self.check_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10  # standard penalty
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10  # standard reward
            self._spawn_food()
        else:
            self.snake.pop()
        
        self._render_frame()
        self.clock.tick(GAME_SPEED)
        
        return reward, game_over, self.score

    def check_collision(self, point=None):
        if point is None:
            point = self.head
        
        #boundary Check
        if (point.x > self.w - BLOCK_SIZE or point.x < 0 or 
            point.y > self.h - BLOCK_SIZE or point.y < 0):
            return True
        
        # self collision Check
        if point in self.snake[1:]:
            return True
        return False

    def _update_direction(self, action):
        # ordering of directions for left/right turning
        clock_wise = [Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Action.RIGHT: x += BLOCK_SIZE
        elif self.direction == Action.LEFT: x -= BLOCK_SIZE
        elif self.direction == Action.DOWN: y += BLOCK_SIZE
        elif self.direction == Action.UP:   y -= BLOCK_SIZE

        self.head = Coords(x, y)

    def _render_frame(self):
        self.display.fill(COLOR_BG)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, COLOR_SNAKE, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            
        pygame.draw.rect(self.display, COLOR_FOOD, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = FONT.render(f"Score: {self.score}", True, COLOR_TEXT)
        self.display.blit(text, [0, 0])
        pygame.display.flip()