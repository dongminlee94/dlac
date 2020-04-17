import gym
import pickle
import numpy as np

# Initialize environment
env = gym.make('HalfCheetah-v2')
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
print('State dimension:', obs_dim)
print('Action dimension:', act_dim)

# Set a random seed
env.seed(0)
np.random.seed(0)

dataset_size = 100000
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
            if temp.shape[0] >= dataset_size:
                break

        obs = next_obs
 
    # Save dataset
    if temp.shape[0] >= dataset_size:
        with open('halfcheetah_dataset_100000.pickle', 'wb') as f:
            pickle.dump(dataset, f)
        break

# Load dataset
with open('halfcheetah_dataset_100000.pickle', 'rb') as f:
    dataset = pickle.load(f)
    dataset = np.array(dataset)
    print(dataset.shape)