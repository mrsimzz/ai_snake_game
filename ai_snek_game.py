import pygame as pg
import random
import numpy as np
import config

"""
Modifications
1. AI able to reset the game and play again
2. Reward system
3. Now we have to give AI the sense of direction as well
4. Current Frame
5. Collision should punish
6. AI actions to be recorded
"""


pg.init()
font = pg.font.SysFont('arial', 25)

# Game Parameters
game_width = config.game_width
game_height = config.game_height
speed = config.speed
block_size = config.block_size
color_dict = config.color_dict


class AI_snek_game():
    def __init__(self) -> None:
        self.width = game_width
        self.height = game_height

        # Pygame Window Settings
        self.display = pg.display.set_mode(size=(self.width, self.height))
        pg.display.set_caption('Snakes')
        self.clock = pg.time.Clock()

        self.game_reset()

    def game_reset(self) -> None:
        # Snake Attributes
        self.block_size = block_size
        self.snake_head = (self.width/2, self.height/2) # Can make it random initially
        self.snake = [self.snake_head, (self.snake_head[0] - self.block_size, self.snake_head[1]), 
                                  (self.snake_head[0] - (2 * self.block_size), self.snake_head[1])]
        self.snake_speed = speed
        self.direction = 1 # Right

        # Game Attributes
        self.score : int = 0
        self.food = None
        self.place_food()
        self.iteration = 0
    
    def place_food(self) -> None:
        x = random.randint(0, (self.width - self.block_size)//self.block_size) * self.block_size
        y = random.randint(0, (self.height - self.block_size)//self.block_size) * self.block_size
        self.food = (x, y)

        if self.food in self.snake:
            self.place_food()

    def collision(self, coordinates = None) -> bool:
        coordinates = self.snake_head if coordinates is None else coordinates
        if coordinates[0] < 0 or coordinates[0] >= self.width or coordinates[1] < 0 or coordinates[1] >= self.height:
            return True
        if coordinates in self.snake[1:]:
            return True
        return False

    def movement(self, ai_move:int) -> None:

        # Understanding AI's Decision
        # Here we will check the current direction and will assign new move based on AI decision
        movement = [1, 4, 2, 3]
        idx = movement.index(self.direction)

        if np.array_equal(ai_move, [1, 0, 0]): # No Move
            new_move = movement[idx]
        elif np.array_equal(ai_move, [0, 1, 0]): # Left turn
            new_idx = (idx + 1) % 4
            new_move = movement[new_idx]
        elif np.array_equal(ai_move, [0, 0, 1]): # Right Turn
            new_idx = (idx - 1) % 4
            new_move = movement[new_idx]
        
        self.direction = new_move

        # Implementing AI's Decision
        new_x = self.snake_head[0]
        new_y = self.snake_head[1]

        if self.direction == 1: # Right
            new_x += self.block_size
        elif self.direction == 2: # Left
            new_x -= self.block_size
        elif self.direction == 3: # Up
            new_y -= self.block_size
        elif self.direction == 4: # Down
            new_y += self.block_size

        self.snake_head = (new_x, new_y)
        self.snake.insert(0, self.snake_head)
    
    def update_ui(self) -> None:
        self.display.fill(color_dict['black'])

        # Snake
        for i in self.snake:
            pg.draw.rect(self.display, color_dict['blue'], pg.Rect(i[0], i[1], self.block_size, self.block_size))
            pg.draw.rect(self.display, color_dict['light_blue'], pg.Rect(i[0] + 4, i[1] + 4, 13, 13))

        # Food
        pg.draw.rect(self.display, color_dict['red'], pg.Rect(self.food[0], self.food[1], self.block_size, self.block_size))
        text = font.render(f"Score : {self.score}", True, color_dict['white'])
        self.display.blit(text, [0, 0])
        
        pg.display.flip()

    def play(self, ai_move) -> int:

        self.iteration +=1

        # User Inputs
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()

        # Removed user action as now it will be controlled by AI
            
        # Movment - now it will take ai_move in account
        self.movement(ai_move)

        # Check Collision
        reward = 0
        if self.collision() or self.iteration > 100*len(self.snake):
            reward = -10
            return True, self.score, reward

        # Food Placement and snake length
        if self.snake_head == self.food:
            self.score += 1
            reward = 10
            self.place_food()
        else:
            self.snake.pop()
        
        self.update_ui()
        self.clock.tick(self.snake_speed)

        return False, self.score, reward