import rlcard
from rlcard.utils import print_card
import torch
import os
import sys
import numpy as np

sys.path.insert(0, './agent')
from DQN import DQN_agent
from Leduc_Human_agent import HumanAgent as HumanAgentLeduc
from Limit_Human_agent import HumanAgent as HumanAgentLimit
import own_models
import argparse

def game(env='leduc'):
    ''' 
    Playing game with the pre-trained agent
    :param string env: Environment, Leduc or Limit Hold'em
    '''
    # Make environment
    if (env == 'limit'):
        env = rlcard.make('limit-holdem', config={'record_action': True})
        agent = DQN_agent(env.state_shape, env.action_num, hidden_layers=[128, 128])
        save_dir = 'own_models/limit_dqn'
        agent.load_networks(torch.load(os.path.join(save_dir, 'model')))
        print(">> Limit Hold'em pre-trained model")
        # Create human agent
        human_agent = HumanAgentLimit(env.action_num)
    else:
        env = rlcard.make('leduc-holdem', config={'record_action': True})
        agent = DQN_agent(env.state_shape, env.action_num, hidden_layers=[128, 128, 128])
        save_dir = 'own_models/leduc_dqn'
        agent.load_networks(torch.load(os.path.join(save_dir, 'model')))
        print(">> Leduc Hold'em pre-trained model")
        # Create human agent
        human_agent = HumanAgentLeduc(env.action_num)


    env.set_agents([human_agent, agent])

    print(">> Start a new game")

    trajectories, payoffs = env.run(is_training=False)
    # If the human does not take the final action, we need to
    # print other players action
    final_state = trajectories[0][-1][-2]
    action_record = final_state['action_record']
    state = final_state['raw_obs']
    _action_list = []
    for i in range(1, len(action_record)+1):
        # if action_record[-i][0] == state['current_player']:
        #     break
        _action_list.insert(0, action_record[-i])
    for pair in _action_list:
        print('>> Player', pair[0], 'chooses', pair[1])

    # Display the card of the agent
    print('===============     DQN Agent    ===============')
    print_card(env.get_perfect_information()['hand_cards'][1])

    # Display the result (number of chips)
    print('===============     Result     ===============')
    if payoffs[0] > 0:
        print('You win {} chips!'.format(payoffs[0]))
    elif payoffs[0] == 0:
        print('It is a tie.')
    else:
        print('You lose {} chips!'.format(-payoffs[0]))
    print('')

    input("Press any key to continue...")

    input(">> End of the game")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="Environment, leduc or limit. Default is leduc.", type=str)
    args = parser.parse_args()
    try:
        game(args.env)
    except: 
        print('Invalid argument')

if __name__ == "__main__":
    main()
