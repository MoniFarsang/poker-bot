import rlcard
from rlcard import models
from rlcard.agents import RandomAgent
from rlcard.utils import seeding, tournament
from rlcard.utils import Logger
import torch
import os
import sys
sys.path.insert(0, './agent')
from DQN import DQN_agent

# Create environments
env = rlcard.make('leduc-holdem', config={'seed': 0})
eval_env = rlcard.make('leduc-holdem', config={'seed': 0})

# Set a global seed
seeding.create_seed(0)

# The paths for saving the logs and learning curves
log_dir = './experiments/limit_holdem_dqn_result/'

# Create DQN agent
agent = DQN_agent(state_no=env.state_shape,
                 act_no=env.action_num, 
                 replay_memory_min_sample=1000,
                 training_period=10,
                 hidden_layers=[128, 128],
                 device=torch.device('cpu'))

# Create a random agent
random_agent = RandomAgent(action_num=eval_env.action_num)

# Create a pre-trained NFSP agent
pretrained_agent = models.load('leduc-holdem-nfsp').agents[0]

# Add the agent to the environments
env.set_agents([agent, random_agent])
# eval_env.set_agents([agent, random_agent])
eval_env.set_agents([agent, pretrained_agent])

# Initialize logger
logger = Logger(log_dir)

# Number of episodes, number of games during evaluation and evaluation in every N steps
episode_no, evaluate_games, evaluate_period = 10000, 1000, 100

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

# Plot the learning curve
logger.plot('DQN')

# Save model
save_dir = 'models/dqn'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
state_dict = agent.get_state_dict()
torch.save(state_dict, os.path.join(save_dir, 'model.pth'))