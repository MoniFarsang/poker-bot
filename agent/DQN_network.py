import numpy as np
import torch
import torch.nn as nn

class DQN_network(object):
    '''
    Deep Q-Network
    '''

    def __init__(self, state_no=36, act_no=4, hidden_layers=[64, 32], learning_rate=0.001, device=None):
        ''' 
        Initilalize the DQN_network object.

        :param act_no (int): Number of actions (4 in Leduc Hold'em)
        :param state_no (list): Size of the state space (36 in Leduc Hold'em)
        :param hidden_layers (list): Dimension of the hidden layers
        :param device (torch.device): Usage CPU or GPU
        '''
        self.state_no = state_no
        self.act_no = act_no
        self.hidden_layers = hidden_layers
        self.learning_rate=learning_rate
        self.device = device

        # DQN network based on the layers
        layers = self.state_no + self.hidden_layers
        DQN_network = [nn.Flatten()]
        DQN_network.append(nn.BatchNorm1d(layers[0]))
        for i in range(len(layers)-1):
            DQN_network.append(nn.Linear(layers[i], layers[i+1], bias=True))
            DQN_network.append(nn.Tanh())
        DQN_network.append(nn.Linear(layers[-1], self.act_no, bias=True))
        DQN_network = nn.Sequential(*DQN_network)

        DQN_network = DQN_network.to(self.device)
        self.DQN_network = DQN_network
        self.DQN_network.eval()

        # Initialize weights in the network
        for p in self.DQN_network.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)

        # Define loss function
        self.loss_function = nn.MSELoss(reduction='mean')

        # Define optimizer
        #self.optimizer =  torch.optim.Adam(self.DQN_network.parameters(), lr=self.learning_rate)
        self.optimizer = torch.optim.RMSprop(self.DQN_network.parameters())


    def get_qvalue(self, next_state_batch):
        ''' 
        Get Q-values for the batch of the next states.
        It does not use gradient calculation.

        :param np.ndarray next_state_batch: Batch of the next states
        :return np.ndarray Q_values: The estimated Q-values
        '''
        # Disable gradient calculation
        with torch.no_grad():
            # Create torch tensor
            next_state_batch = torch.from_numpy(next_state_batch).float().to(self.device)
            # Get Q values
            Q_values = self.DQN_network(next_state_batch).cpu().numpy()
        return Q_values

    def update(self, state_batch, action_batch, target_batch):
        ''' 
        Update the policy network

        :param np.ndarray state_batch: Batch of states from replay memory
        :param np.ndarray action_batch: Batch of actions from replay memory
        :param np.ndarray target_batch: Batch of Q-values from the target policy, it used during the optimization step
        :return float batch_loss: The calculated loss on the batch       
        '''
        # Set the gradients to zero
        self.optimizer.zero_grad()

        # Set the network in training mode
        self.DQN_network.train()

        # Create torch tensors
        state_batch = torch.from_numpy(state_batch).float().to(self.device)
        action_batch = torch.from_numpy(action_batch).long().to(self.device)
        target_batch = torch.from_numpy(target_batch).float().to(self.device)

        # Gather Q-values from network and replay memory actions
        Q_values = torch.gather(self.DQN_network(state_batch), dim=-1, index=action_batch.unsqueeze(-1)).squeeze(-1)

        # Optimization step
        batch_loss = self.loss_function(Q_values, target_batch)
        batch_loss.backward()
        self.optimizer.step()
        batch_loss = batch_loss.item()
        self.DQN_network.eval()
        return batch_loss
