import os
import gym
import time
import argparse
import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Configurations
parser = argparse.ArgumentParser(description='RL algorithms with PyTorch in MuJoCo environments')
parser.add_argument('--env', type=str, default='HalfCheetah-v2', 
                    help='choose an environment between Hopper-v2 and HalfCheetah-v2')
parser.add_argument('--path', type=str, default=None, 
                    help='path to load the trained embedding model')
parser.add_argument('--algo', type=str, default='sac', 
                    help='select an algorithm between ppo and sac')
parser.add_argument('--mode', type=str, default='embed',   # 'embed' or 'raw'
                    help='select an mode between embedded data and raw data')
parser.add_argument('--seed', type=int, default=0, 
                    help='seed for random number generators')
parser.add_argument('--iterations', type=int, default=200, 
                    help='iterations to run and train agent')
parser.add_argument('--steps_per_iter', type=int, default=5000, 
                    help='steps of interaction for the agent and the environment in each epoch')
parser.add_argument('--max_step', type=int, default=1000,
                    help='max episode step')
parser.add_argument('--gpu_index', type=int, default=0, metavar='N')
args = parser.parse_args()
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

if args.algo == 'ppo':
    from agents.ppo import Agent
elif args.algo == 'sac':
    from agents.sac import Agent

def main():
    """Main."""
    # Initialize environment
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    print('State dimension:', obs_dim)
    print('Action dimension:', act_dim)

    # Set a random seed
    env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create an agent
    if args.algo == 'sac':                                                        
        agent = Agent(env, args, device, obs_dim, act_dim, act_limit, 
                      hidden_sizes=(300,300), buffer_size=int(1e6), batch_size=100, 
                      alpha=0.2, actor_lr=1e-4, qf_lr=1e-3)   
    elif args.algo == 'ppo':
        agent = Agent(env, args, device, obs_dim, act_dim, act_limit, sample_size=4000)

    # Create a SummaryWriter object by TensorBoard
    dir_name = 'runs/' + args.env + '/' \
                                  + args.algo \
                                  + '_' + args.mode \
                                  + '_hs_300_alr_1e-4_clr_1e-3_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    writer = SummaryWriter(log_dir=dir_name)

    start_time = time.time()

    total_num_steps = 0
    train_sum_returns = 0.
    train_num_episodes = 0

    # Main loop
    for i in range(args.iterations):
        train_step_count = 0
        while train_step_count <= args.steps_per_iter:
            # Perform the training phase, during which the agent learns
            agent.eval_mode = False
            
            # Run one episode
            train_step_length, train_episode_return = agent.run(args.max_step)
            
            total_num_steps += train_step_length
            train_step_count += train_step_length
            train_sum_returns += train_episode_return
            train_num_episodes += 1

            train_average_return = train_sum_returns / train_num_episodes if train_num_episodes > 0 else 0.0

            # Log experiment result for training steps
            writer.add_scalar('Train/AverageReturns', train_average_return, total_num_steps)
            writer.add_scalar('Train/EpisodeReturns', train_episode_return, total_num_steps)

        # Perform the evaluation phase -- no learning
        agent.eval_mode = True
        
        eval_sum_returns = 0.
        eval_num_episodes = 0

        for _ in range(10):
            # Run one episode
            eval_step_length, eval_episode_return = agent.run(args.max_step)

            eval_sum_returns += eval_episode_return
            eval_num_episodes += 1

        eval_average_return = eval_sum_returns / eval_num_episodes if eval_num_episodes > 0 else 0.0

        # Log experiment result for evaluation steps
        writer.add_scalar('Eval/AverageReturns', eval_average_return, total_num_steps)
        writer.add_scalar('Eval/EpisodeReturns', eval_episode_return, total_num_steps)

        print('---------------------------------------')
        print('Iterations:', i)
        print('Steps:', total_num_steps)
        print('Episodes:', train_num_episodes)
        print('AverageReturn:', round(train_average_return, 2))
        print('EvalEpisodes:', eval_num_episodes)
        print('EvalAverageReturn:', round(eval_average_return, 2))
        print('OtherLogs:', agent.logger)
        print('Time:', int(time.time() - start_time))
        print('---------------------------------------')

        # Save a training model
        # if (i > 0) and (i % 10 == 0):
        #     if not os.path.exists('./asset'):
        #         os.mkdir('./asset')
            
        #     ckpt_path = os.path.join('./asset/' + args.env + '_' + args.algo \
        #                                                    + '_' + args.mode \
        #                                                    + '_i_' + str(i) \
        #                                                    + '_st_' + str(total_num_steps) \
        #                                                    + '_tr_' + str(round(train_average_return, 2)) \
        #                                                    + '_er_' + str(round(eval_average_return, 2)) \
        #                                                    + '_t_' + str(int(time.time() - start_time)) + '.pt')
            
        #     torch.save(agent.actor.state_dict(), ckpt_path)

if __name__ == "__main__":
    main()
