# AI Snake Game Using Reinforcement Learning

This project is an implementation of the classic Snake game with an AI agent trained using reinforcement learning. The project is organized into four main files:

- **`agent.py`**: Contains the reinforcement learning agent, including the Q-learning algorithm and the logic for training the agent.
- **`model.py`**: Defines the neural network model used by the agent to predict the best actions based on the current state of the game.
- **`config.py`**: Houses global parameters and configurations such as game dimensions, speed, and training hyperparameters.
- **`ai_snek_game.py`**: Manages the game environment, including the game loop, state representation, and interactions between the snake and its environment.
- **`model_folder`**: Stores the latest trained model’s state dictionary, which can be loaded to resume training or for inference.

The AI agent learns to play the game by interacting with the environment, receiving rewards for eating food, and penalties for hitting walls or itself. Over time, it improves its strategy and aims to achieve the highest possible score.

---

You can customize this further based on specific details or features of your implementation.
