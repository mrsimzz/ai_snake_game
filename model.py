from fileinput import filename
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

class linear_qnet(nn.Module):
    def __init__(self, input_size : int, hidden_size : int, output_size : int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model_folder'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class qtrainer():
    def __init__(self, model : nn.Module, lr : float, gamma : float) -> None:
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.loss_funtion = nn.MSELoss()
    
    def converison_to_tensor(self, input):
        return torch.tensor(input, dtype=torch.float)

    def train_step(self, state, action, reward, next_state, game_over):
        # Conversion to pytorch tensor
        state = self.converison_to_tensor(state)
        action = self.converison_to_tensor(action)
        reward = self.converison_to_tensor(reward)
        next_state = self.converison_to_tensor(next_state)

        # Handling multiple sizes
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            game_over = (game_over, )
        
        # Predicted q values with current state, inference_mode to make the predictions faster and pytorch don't store unnecessary gradients
        self.model.train()
        pred = self.model(state)

        # pred.clone()
        # Below is the bellman
        # q_new = r + (γ * max(next_predited) q value) -> only do this if not done
        # preds[argmax[action]] = q_new

        target = pred.clone()
        
        # Implementation of q-learning
        for idx in range(len(game_over)):
            q_new = reward[idx]
            if not game_over[idx]:
                q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            target[idx][torch.argmax(action).item()] = q_new

        # To empty the gradient
        self.optimizer.zero_grad()

        # Calculate the loss
        loss = self.loss_funtion(target, pred)

        # Appling backpropogation
        loss.backward()

        # Optimizing (Performing gd)
        self.optimizer.step()