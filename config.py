
# This file contains all the global parameters

# File : ai_snek_game.py
# Game Parameters
game_width : int = 480
game_height : int = 480
speed : int = 50
block_size : int = 20

color_dict : dict = {
    'white' : (255, 255, 255),
    'black' : (0, 0, 0),
    'red' : (200, 0, 0),
    'blue' : (0, 0, 255),
    'light_blue' : (0, 100, 255)
}

# File : agent.py
max_memory = 100_000
block_size = 20
batch_size = 1000
learning_rate = 0.01