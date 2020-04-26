import os
import gym
import time
import argparse
import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agent.sac import Agent

# Configurations
parser = argparse.ArgumentParser(description='SAC algorithm with PyTorch in LunarLanderContinuous environment')
parser.add_argument('--path', type=str, default=None, 
                    help='path to load the trained embedding model')
parser.add_argument('--mode', type=str, default='raw',   # 'embed' or 'raw'
                    help='select an mode between embedded data and raw data')
parser.add_argument('--training_eps', type=int, default=1000, 
                    help='training episode number')
parser.add_argument('--eval_per_train', type=int, default=100, 
                    help='evaluation number per training')
parser.add_argument('--evaluation_eps', type=int, default=100,
                    help='evaluation episode number')
parser.add_argument('--gpu_index', type=int, default=0, metavar='N')
args = parser.parse_args()
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')


def main():
    """Main."""
    # Initialize environment
    env = gym.make('LunarLanderContinuous-v2')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    print('State dimension:', obs_dim)
    print('Action dimension:', act_dim)

    # Set a random seed
    env.seed(40)
    np.random.seed(40)
    torch.manual_seed(40)

    # Create an agent
    agent = Agent(env, args, device, obs_dim, act_dim, act_limit, 
                  hidden_sizes=(300,300), buffer_size=int(1e6), batch_size=100, 
                  alpha=0.3, actor_lr=1e-3, qf_lr=1e-3)

    # Create a SummaryWriter object by TensorBoard
    dir_name = 'runs/' + 'LunarLanderContinuous/' + args.mode + '_hs_300_alr_1e-3_clr_1e-3'
    writer = SummaryWriter(log_dir=dir_name)

    start_time = time.time()

    train_num_steps = 0
    train_sum_returns = 0.
    train_num_episodes = 0

    # Runs a full experiment, spread over multiple training episodes
    for episode in range(args.training_eps):
        # Perform the training phase, during which the agent learns
        agent.eval_mode = False
        
        # Run one episode
        train_step_length, train_episode_return = agent.run(env._max_episode_steps)
        
        train_num_steps += train_step_length
        train_sum_returns += train_episode_return
        train_num_episodes += 1

        train_average_return = train_sum_returns / train_num_episodes if train_num_episodes > 0 else 0.0

        # Log experiment result for training episodes
        writer.add_scalar('Train/AverageReturns', train_average_return, episode)
        writer.add_scalar('Train/EpisodeReturns', train_episode_return, episode)

        # Perform the evaluation phase -- no learning
        if (episode + 1) % args.eval_per_train == 0:
            agent.eval_mode = True
            
            eval_sum_returns = 0.
            eval_num_episodes = 0

            for _ in range(args.evaluation_eps):
                # Run one episode
                eval_step_length, eval_episode_return = agent.run(env._max_episode_steps)

                eval_sum_returns += eval_episode_return
                eval_num_episodes += 1

                eval_average_return = eval_sum_returns / eval_num_episodes if eval_num_episodes > 0 else 0.0

                # Log experiment result for evaluation episodes
                writer.add_scalar('Eval/AverageReturns', eval_average_return, episode)
                writer.add_scalar('Eval/EpisodeReturns', eval_episode_return, episode)

            print('---------------------------------------')
            print('Steps:', train_num_steps)
            print('Episodes:', train_num_episodes)
            print('AverageReturn:', round(train_average_return, 2))
            print('EvalEpisodes:', eval_num_episodes)
            print('EvalAverageReturn:', round(eval_average_return, 2))
            print('OtherLogs:', agent.logger)
            print('Time:', int(time.time() - start_time))
            print('---------------------------------------')

if __name__ == "__main__":
    main()
