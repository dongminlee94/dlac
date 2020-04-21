import os
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


# Configurations
parser = argparse.ArgumentParser(description='VAE Embedding')
parser.add_argument('--env', type=str, default='Hopper-v2', 
                    help='choose an environment between Hopper-v2 and HalfCheetah-v2')
parser.add_argument('--path', type=str, default=None, 
                    help='path to load the dataset')
parser.add_argument('--seed', type=int, default=0, 
                    help='seed for random number generators (default: 0)')
parser.add_argument('--epochs', type=int, default=50, 
                    help='number of epochs to train (default: 50)')
parser.add_argument('--batch_size', type=int, default=128, 
                    help='input batch size for training (default: 128)')
parser.add_argument('--gpu_index', type=int, default=0, metavar='N')
args = parser.parse_args()
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

def main():
    # Set the shapes of observation space and action space in the environment
    if args.env == 'Hopper-v2':
        obs_dim = 11
        act_dim = 3
    elif args.env == 'HalfCheetah-v2':
        obs_dim = 17
        act_dim = 6

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
                                  + 'ds_' + str(np.array(dataset).shape[0]) \
                                  + '_ep_' + str(args.epochs) \
                                  + '_t_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    writer = SummaryWriter(log_dir=dir_name)

    start_time = time.time()
    
    num_count = 0
    sum_losses = 0.
    sum_reconst = 0.
    sum_kld = 0.

    # Start training
    for epoch in range(args.epochs):
        for i, (obs, action, next_obs) in enumerate(data_loader):
            if 0: # Check shape of transition experiences
                print(obs.shape)
                print(action.shape)
                print(next_obs.shape)
            
            pred_next_obs, mu, logvar = model(obs.to(device).float(), action.to(device))

            # Compute reconstruction loss and kl divergence
            reconst = F.mse_loss(pred_next_obs, next_obs.to(device).float(), size_average=False)
            # For KL divergence, see Appendix B from VAE paper: https://arxiv.org/abs/1312.6114
            # - 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kld = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).to(device)

            # Update VAE model parameters
            loss = reconst + 1e-2 * kld
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute average losses
            num_count += 1
            sum_losses += loss.item()
            sum_reconst += reconst.item()
            sum_kld += kld.item()

            average_loss = sum_losses / num_count
            average_reconst = sum_reconst / num_count
            average_kld = sum_kld / num_count

            if (i + 1) % 100 == 0:
                print('---------------------------------------')
                print('Epoch: [{}/{}]'.format(epoch+1, args.epochs))
                print('Step: [{}/{}]'.format(i+1, len(data_loader)))
                print('AverageLoss: {:.4f}'.format(average_loss))
                print('Loss: {:.4f}'.format(loss.item()))
                print('AverageReconst: {:.4f}'.format(average_reconst))
                print('Reconst: {:.4f}'.format(reconst.item()))
                print('AverageKL: {:.4f}'.format(average_kld))
                print('KL: {:.4f}'.format(kld.item()))
                print('Time:', int(time.time() - start_time))
                print('---------------------------------------')
    
        # Log experiment result for training steps
        writer.add_scalar('Train/AverageLoss', average_loss, epoch)
        writer.add_scalar('Train/EpochLoss', loss, epoch)
        writer.add_scalar('Train/AverageReconst', average_reconst, epoch)
        writer.add_scalar('Train/EpochReconst', reconst, epoch)
        writer.add_scalar('Train/AverageKL', average_kld, epoch)
        writer.add_scalar('Train/EpochKL', kld, epoch)

    # Save the trained model
    if not os.path.exists('../asset'):
        os.mkdir('../asset')
    
    ckpt_path = os.path.join('../asset/' + args.env \
                                         + '_ds_' + str(np.array(dataset).shape[0]) \
                                         + '_ep_' + str(args.epochs) \
                                         + '_al_' + str(round(average_loss, 2)) \
                                         + '_el_' + str(round(loss.item(), 2)) \
                                         + '_t_' + str(int(time.time() - start_time)) 
                                         + '.pt')
    
    torch.save(model.state_dict(), ckpt_path)

if __name__ == "__main__":
    main()