import gym
import torch
import pickle
import argparse
import numpy as np

# Configurations
parser = argparse.ArgumentParser(description='Make a dataset from a uniformly random policy')
parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2', 
                    help='choose an environment between LunarLanderContinuous-v2 and Hopper-v2')
parser.add_argument('--name', type=str, default=None,
                    help='name to save the dataset')
parser.add_argument('--d_size', type=int, default=100000,
                    help='dataset size')
args = parser.parse_args()

# Initialize environment
env = gym.make(args.env)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
print('State dimension:', obs_dim)
print('Action dimension:', act_dim)

# Set a random seed
env.seed(40)
np.random.seed(40)

dataset = []

while True:
    obs = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)

        if not done:
            dataset.append((obs, action, next_obs))
            temp = np.array(dataset)
            if temp.shape[0] >= args.d_size:
                break

        obs = next_obs
 
    # Save dataset
    if temp.shape[0] >= args.d_size:
        pickle.dump(dataset, open(args.name, "wb"))
        break

# Load dataset
dataset = pickle.load(open(args.name, "rb"))
dataset = np.array(dataset)
print(dataset.shape)