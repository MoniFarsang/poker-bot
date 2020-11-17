import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import random

from ReplayMemory import ReplayMemory
from DQN_network import DQN_network

class DQN_agent(object):
    '''
    DQN agent
    '''
    def __init__(self,
                state_no,
                act_no,
                extra_action_version=0,
                replay_memory_capacity=20000,
                replay_memory_min_sample=1000,
                batch_size=32,
                training_period=1,
                discount_factor=0.99,
                hidden_layers=[64, 32],
                learning_rate=0.0001,
                epsilon_decay_steps=20000,
                update_target_dqn_period=1000, 
                device=None):

        '''
        Initialize the DQN agent

        :param int state_no: Number of states
        :param int act_no: Number of actions
        :param int extra_action_version: Mode of choosing action during evaluation phase. Action with maximum value: 0, Raise action instead of Call if possible: 1, Raise action instead of Check if possible: 2, Raise action instead of Fold if possible: 3
        :param int replay_memory_capacity: Replay memory size
        :param int replay_memory_min_sample: Minimum number of samples in the replay memory during sampling
        :param int batch_size: Size of batches to sample from the replay memory
        :param int training_period: Train the network in every N steps
        :param float discount_factor: Discount factor (gamma) during training the agent
        :param list[int] hidden_layers: Dimensions of the hidden layers in the DQN network
        :param float learning_rate: The learning rate in the DQN network
        :param int epsilon_decay_steps: Number of steps to decay epsilon
        :param int update_target_dqn_period: Update target network in every N steps
        :param torch.device device: Usage CPU or GPU
        '''
        
        self.replay_memory_min_sample = replay_memory_min_sample
        self.update_target_dqn_period = update_target_dqn_period
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.act_no = act_no
        self.training_period = training_period
        self.extra_action_version = extra_action_version

        # Torch device on which a torch.Tensor will be allocated
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Create the replay memory
        self.memory = ReplayMemory(replay_memory_capacity, batch_size)

        # Initialize current timestep and current training timestep
        self.current_timestep, self.current_training_timestep = 0, 0

        # Create array for the epsilon values during the epsilon decay 
        self.epsilons = np.linspace(1.0, 0.1, epsilon_decay_steps)

        # Create the policy and the target network
        self.policy_dqn = DQN_network(act_no=act_no, learning_rate=learning_rate, state_no=state_no, hidden_layers=hidden_layers, device=self.device)
        self.target_dqn = DQN_network(act_no=act_no, learning_rate=learning_rate, state_no=state_no, hidden_layers=hidden_layers, device=self.device)

        # Set use_raw value for the RLCard environment
        self.use_raw = False

    def store_and_train(self, transition):
        ''' 
        Save transition into memory and train the agent based on the training period.

        :param tuple transition: The transition tuple 'state', 'action', 'reward', 'next_state', 'done'
        
        '''
        (state, action, reward, next_state, done) = tuple(transition)

        # Store transition in replay memory
        self.memory.push(state['obs'], action, reward, next_state['obs'], done)
        # Increment the number of timesteps
        self.current_timestep += 1
        # Train the agent if the replay memory has data already and agent reached the next training period
        time_between = self.current_timestep - self.replay_memory_min_sample
        if time_between>=0 and time_between%self.training_period == 0:
            self.train()

    def discard_invalid_actions(self, action_probs, valid_actions):
        ''' 
        Remove invalid actions and normalize the probabilities.

        :param numpy.array[float] action_probs: Probabilities of all action
        :param list[int] valid_actions: Valid actions in the current state
        :return numpy.array[float] norm_valid_action_probs: Probabilities of valid actions
        '''
        # Initialize new array
        norm_valid_action_probs = np.zeros(action_probs.shape[0])
        # Add probability values of valid actions to the array
        norm_valid_action_probs[valid_actions] = action_probs[valid_actions]
        # Normalize probabilities
        norm_valid_action_probs[valid_actions] = 1 / len(valid_actions)
        return norm_valid_action_probs

    def predict(self, state):
        ''' 
        Predict the action probabilities.

        :param numpy.array[float] state: Current state
        :return numpy.array[float] q_values: Array of Q values  
        '''
        epsilon = self.epsilons[min(self.current_timestep, self.epsilon_decay_steps-1)]
        actions = np.ones(self.act_no, dtype=float) * epsilon / self.act_no
        q_values = self.policy_dqn.get_qvalue(np.expand_dims(state, 0))[0]
        best_action = np.argmax(q_values)
        actions[best_action] += (1.0 - epsilon)
        return actions

    def step(self, state):
        ''' 
        Define step function for the RLCard environment.
        Get the action for the current state for training purpose.
        If neccessary, remove invalid action pobabilities.

        :param numpy.array state: The current state
        :return int action: The chosen action in the current state
        '''
        actions = self.predict(state['obs'])
        norm_valid_action_probs = self.discard_invalid_actions(actions, state['legal_actions'])
        action = np.random.choice(np.arange(len(actions)), p=norm_valid_action_probs)
        return action


    def eval_step(self, state):
        ''' 
        Define eval_step function for the RLCard environment.
        Get the action for the evaluation purpose instead of training purpose.

        :param numpy.array state: The current state
        :return int action: The chosen action in the current state
        '''
        q_values = self.policy_dqn.get_qvalue(np.expand_dims(state['obs'], 0))[0]
        norm_valid_action_probs = self.discard_invalid_actions(np.exp(q_values), state['legal_actions'])
        # Check version of choosing action
        if self.extra_action_version == 1:
          # If Raise (1) is a valid action and the best action is Call (0)
          if 1 in state['legal_actions'] and np.argmax(norm_valid_action_probs)==0:
            best_action = 1
          else:
            best_action = np.argmax(norm_valid_action_probs)
        elif self.extra_action_version == 2:
          # If Raise (1) is a valid action and the best action is Check (3)
          if 1 in state['legal_actions'] and np.argmax(norm_valid_action_probs)==3:
            best_action = 1
          else:
            best_action = np.argmax(norm_valid_action_probs)
        elif self.extra_action_version == 3:
          # If Raise (1) is a valid action and the best action is Fold (2)
          if 1 in state['legal_actions'] and np.argmax(norm_valid_action_probs)==2:
            best_action = 1
          else:
            best_action = np.argmax(norm_valid_action_probs)
        else:
          best_action = np.argmax(norm_valid_action_probs)
        return best_action, norm_valid_action_probs

    
    def train(self):
        ''' 
        Train the agent.

        return float loss: The loss of the current batch
        '''
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)

        # Get best next action using the policy network
        q_values_next = self.policy_dqn.get_qvalue(next_state_batch)
        best_actions = np.argmax(q_values_next, axis=1)

        # Calculate Q values from the target policy
        q_values_next_target = self.target_dqn.get_qvalue(next_state_batch)
        target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * self.discount_factor * q_values_next_target[np.arange(self.batch_size), best_actions]

        # Update policy network
        state_batch = np.array(state_batch)
        loss = self.policy_dqn.update(state_batch, action_batch, target_batch)

        # Update target network based on the target update period
        if self.current_training_timestep % self.update_target_dqn_period == 0:
            self.target_dqn = deepcopy(self.policy_dqn)

        self.current_training_timestep += 1


    def get_state_dict(self):
        ''' 
        Get the state dictionaries.

        :return dict model_dict: Dictionaries containing the whole state of the policy and target modules
        '''
        model_dict = {'policy_network': self.policy_dqn.DQN_network.state_dict(), 'target_network': self.target_dqn.DQN_network.state_dict()}
        return model_dict

    def load_networks(self, checkpoint):
        ''' 
        Load network models.

        :param dict checkpoint: Checkpoint of the policy and target networks
        '''
        self.policy_dqn.DQN_network.load_state_dict(checkpoint['policy_network'])
        self.target_dqn.DQN_network.load_state_dict(checkpoint['target_network'])