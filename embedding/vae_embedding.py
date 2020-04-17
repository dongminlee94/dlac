import os
import time
import pickle
import argparse
import datetime
import numpy as np

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Configurations
parser = argparse.ArgumentParser(description='VAE Embedding')
parser.add_argument('--seed', type=int, default=0, 
                    help='seed for random number generators')
parser.add_argument('--epochs', type=int, default=50, 
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=128, 
                    help='input batch size for training')
args = parser.parse_args()

def main():
    # Set a random seed
    env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load dataset
    dataset = pickle.load(open('../dataset/dataset.pickle', "rb"))
    dataset = np.array(dataset)
    print(dataset.shape)



if __name__ == "__main__":
    main()