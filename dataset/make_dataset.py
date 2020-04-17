import gym
import pickle
import argparse
import numpy as np

# Configurations
parser = argparse.ArgumentParser(description='Make a dataset in MuJoCo environment from a uniformly random policy')
parser.add_argument('--path', type=str, default=None,
                    help='path to save the dataset')
parser.add_argument('--d_size', type=int, default=100000,
                    help='dataset size')
args = parser.parse_args()

# Initialize environment
env = gym.make('Hopper-v2')
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
print('State dimension:', obs_dim)
print('Action dimension:', act_dim)

# Set a random seed
env.seed(0)
np.random.seed(0)

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
        pickle.dump(dataset, open(args.path, "wb"))
        break

# Load dataset
dataset = pickle.load(open(args.path, "rb"))
dataset = np.array(dataset)
print(dataset.shape)