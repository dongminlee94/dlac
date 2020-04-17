import os
import gym
import time
import pickle
import argparse
import datetime
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model import VAE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Configurations
parser = argparse.ArgumentParser(description='VAE Embedding')
parser.add_argument('--env', type=str, default='Hopper-v2', 
                    help='choose an environment between Hopper-v2 and HalfCheetah-v2')
parser.add_argument('--path', type=str, default=None, 
                    help='path to load the dataset')
parser.add_argument('--seed', type=int, default=0, 
                    help='seed for random number generators')
parser.add_argument('--epochs', type=int, default=50, 
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=128, 
                    help='input batch size for training')
args = parser.parse_args()

def main():
    # Initialize an environment
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print('State dimension:', obs_dim)
    print('Action dimension:', act_dim)

    # Set a random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load dataset
    dataset = pickle.load(open(args.path, "rb"))

    # Set Data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              drop_last=True)

    # Set VAE model and an optimizer
    model = VAE(obs_dim, obs_dim, act_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Create a SummaryWriter object by TensorBoard
    dir_name = 'runs/' + args.env + '/' \
                + '_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    writer = SummaryWriter(log_dir=dir_name)

    start_time = time.time()

    # Start training
    for epoch in range(args.epochs):
        for obs, action, next_obs in data_loader:
            if 0: # Check shape of transition experiences
                print(obs.shape)
                print(action.shape)
                print(next_obs.shape)
            
            pred_next_obs, mu, logvar = model(obs.float(), action)

            # Compute reconstruction loss and kl divergence
            reconst_loss = F.mse_loss(pred_next_obs, next_obs.float(), size_average=False)
            # For KL divergence, see Appendix B from VAE paper: https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kld = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).to(device)

            # Update VAE model parameters
            loss = reconst_loss + 1e-4 * kld
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save losses
    
        # Log experiment result for training steps
    #     writer.add_scalar('Train/AverageReturns', train_average_return, total_num_steps)
    #     writer.add_scalar('Train/EpisodeReturns', train_episode_return, total_num_steps)

    #     print('---------------------------------------')
    #     print('Iterations:', i)
    #     print('Steps:', total_num_steps)
    #     print('Episodes:', train_num_episodes)
    #     print('AverageReturn:', round(train_average_return, 2))
    #     print('EvalEpisodes:', eval_num_episodes)
    #     print('EvalAverageReturn:', round(eval_average_return, 2))
    #     print('OtherLogs:', agent.logger)
    #     print('Time:', int(time.time() - start_time))
    #     print('---------------------------------------')

    # # Save a training model
    # if (i > 0) and (i % 10 == 0):
    #     if not os.path.exists('./tests/save_model'):
    #         os.mkdir('./tests/save_model')
        
    #     ckpt_path = os.path.join('./tests/save_model/' + args.env + '_' + args.algo \
    #                                                                     + '_i_' + str(i) \
    #                                                                     + '_st_' + str(total_num_steps) \
    #                                                                     + '_tr_' + str(round(train_average_return, 2)) \
    #                                                                     + '_er_' + str(round(eval_average_return, 2)) \
    #                                                                     + '_t_' + str(int(time.time() - start_time)) + '.pt')
        
    #     torch.save(agent.actor.state_dict(), ckpt_path)

if __name__ == "__main__":
    main()