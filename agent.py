import numpy as np
import random
from sympy import true
import torch
import matplotlib.pyplot as plt
from IPython import display
from collections import deque

from ai_snek_game import AI_snek_game
from model import linear_qnet, qtrainer
import config

max_memory = config.max_memory
block_size = config.block_size
batch_size = config.batch_size
learning_rate = config.learning_rate

plt.ion()

class Agent007():

    def __init__(self) -> None:
        self.n_games = 0
        self.epsolon = 0 # Randomness
        self.gamma = 0.9 # Discount rate
        self.memory = deque(maxlen = max_memory)

        # Models and trainer
        self.model = linear_qnet(11, 256, 3)
        self.trainer = qtrainer(self.model, lr=learning_rate, gamma=self.gamma)

    def get_state(self, game) -> np.ndarray:
        head = game.snake_head

        # Awareness around the head
        a_left = (head[0] - block_size, head[1])
        a_right = (head[0] + block_size, head[1])
        a_down = (head[0], head[1] + block_size)
        a_up = (head[0], head[1] - block_size)

        # Check Current Direction
        dir_r = game.direction == 1 # Right
        dir_l = game.direction == 2 # Left
        dir_u = game.direction == 3 # Up
        dir_d = game.direction == 4 # Down
        
        states = [
            # Danger Straight
            (dir_r and game.collision(a_right)) or
            (dir_l and game.collision(a_left)) or
            (dir_u and game.collision(a_up)) or
            (dir_d and game.collision(a_down)),

            # Danger Left
            (dir_r and game.collision(a_up)) or
            (dir_l and game.collision(a_down)) or
            (dir_u and game.collision(a_left)) or
            (dir_d and game.collision(a_right)),

            # Danger Right
            (dir_r and game.collision(a_down)) or
            (dir_l and game.collision(a_up)) or
            (dir_u and game.collision(a_right)) or
            (dir_d and game.collision(a_left)),

            # Current Movement
            dir_l, dir_r, dir_u, dir_d,

            # Food Location
            game.food[0] > head[0], # Food Right
            game.food[0] < head[0], # Food Left
            game.food[1] > head[1], # Food Down
            game.food[1] < head[1] # Food Up
        ]

        return np.array(states, dtype = int)

    def store(self, state, action, reward, next_state, game_over) -> None:
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_memory(self) -> None:
        # Batch Training
        if len(self.memory) > batch_size:
            mini = random.sample(self.memory, batch_size)
        else:
            mini = self.memory
        
        states, actions, rewards, next_states, game_overs = zip(*mini, strict = True)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over) -> None:
        self.trainer.train_step(state, action, reward, next_state, game_over)


    # Do additional research
    def get_action(self, state:np.ndarray) -> list[int]:
        # Random moves : Tradeoff exploration/exploitation
        self.epsolon = 80 - self.n_games
        final_move = [0, 0, 0]

        if random.randint(0,200) < self.epsolon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 : torch.Tensor = torch.tensor(state, dtype = torch.float)
            self.model.eval()
            with torch.inference_mode():
                prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def plot(scores, mean_scores) -> None:
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

def train() -> None:
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent007()
    game = AI_snek_game()
    
    while True:
        # get old state
        old_state = agent.get_state(game)

        # get move
        final_move = agent.get_action(old_state)

        # perform move and get new state
        game_over, score, reward = game.play(final_move)

        # get new state
        new_state = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state = old_state, 
                                 action = final_move, 
                                 reward = reward, 
                                 next_state = new_state, 
                                 game_over = game_over)

        # store
        agent.store(state = old_state, 
                    action = final_move, 
                    reward = reward, 
                    next_state = new_state, 
                    game_over = game_over)

        if game_over:
            # train long memory and plot the results after each game
            game.game_reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            # Plot
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()