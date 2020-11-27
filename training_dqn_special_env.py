import special_environment
from rlcard import models
from rlcard.agents import RandomAgent
from rlcard.utils import seeding, tournament
from rlcard.utils import Logger
import torch
import os
import sys
import csv
import numpy as np

sys.path.insert(0, './agent')
from DQN import DQN_agent

# Create environments
env = special_environment.make('leduc-holdem', config={'seed': 0})
eval_env = special_environment.make('leduc-holdem', config={'seed': 0})

# Set a global seed
seeding.create_seed(0)

# Play agressive game based on the version of choosing actual action
# Action with maximum value: 0
# Raise action instead of Call if possible: 1
# Raise action instead of Check if possible: 2
# Raise action instead of Fold if possible: 3
extra_action_version=1

# Opponent agent
# Random agent: 0
# Pretrained agent with nfsp: 1
opponent_agent_version_train=1
opponent_agent_version_eval=0

# The paths for saving the logs and learning curves
log_dir = './experiments/leduc_holdem_dqn_result/'

# Create DQN agent
agent = DQN_agent(state_no=env.state_shape,
                  act_no=env.action_num, 
                  replay_memory_min_sample=1000,
                  training_period=10,
                  hidden_layers=[128, 128],
                  device=torch.device('cpu'),
                  extra_action_version=extra_action_version)

# Create opponent agent for training
if opponent_agent_version_train == 1:
  # Create a pre-trained NFSP agent
  opponent_agent_train = models.load('leduc-holdem-nfsp').agents[0]
else:
  # Create a random agent
  opponent_agent_train = RandomAgent(action_num=eval_env.action_num)

# Create opponent agent for evaluation
if opponent_agent_version_eval == 1:
  # Create a pre-trained NFSP agent
  opponent_agent_eval = models.load('leduc-holdem-nfsp').agents[0]
else:
  # Create a random agent
  opponent_agent_eval = RandomAgent(action_num=eval_env.action_num)

# Add the agent to the environments
env.set_agents([agent, opponent_agent_train])
eval_env.set_agents([agent, opponent_agent_eval])

# Initialize logger
logger = Logger(log_dir)

# Number of episodes, number of games during evaluation and evaluation in every N steps
episode_no, evaluate_games, evaluate_period = 1000, 100, 10

for episode in range(episode_no):
    # Generate data from the environment
    trajectories, _ = env.run(is_training=True)

    # Feed transitions into agent memory, and train the agent
    for ts in trajectories[0]:
        agent.store_and_train(ts)

    # Evaluate the performance
    if episode % evaluate_period == 0:
        logger.log_performance(env.timestep, tournament(eval_env, evaluate_games)[0])

# Close files in the logger
logger.close_files()

# Save model
save_dir = 'models/dqn'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
state_dict = agent.get_state_dict()
torch.save(state_dict, os.path.join(save_dir, 'model.pth'))

csv_path = os.path.join(log_dir, 'performance.csv')
save_path = log_dir


def plot(algorithm):
    ''' 
    Read data from csv file and plot the results
    '''
    import matplotlib.pyplot as plt
    with open(csv_path) as csvfile:
        print(csv_path)
        reader = csv.DictReader(csvfile)
        xs = []
        ys = []
        for row in reader:
            xs.append(int(row['timestep']))
            ys.append(float(row['reward']))
        fig, ax = plt.subplots()
        
        # Calculate the trendline
        # z = np.polyfit(xs, ys, 10)
        # p = np.poly1d(z)
        # ax.plot(xs, p(xs))
        ax.plot(xs, ys)

        #ax.plot(xs[:-(N-1)], moving_aves, label=algorithm)
        ax.set(xlabel='timestep', ylabel='reward', title=algorithm)
        ax.legend('DQN')
        ax.grid()

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig.savefig(save_path)

title = 'Leduc Holdem DQN action version: ' + str(extra_action_version) + ', agent training: ' + str(opponent_agent_version_train) + ', agent eval: ' + str(opponent_agent_version_eval)
# Plot the learning curve
plot(title)
