import gym
import numpy as np
import matplotlib.pyplot as plot
from collections import defaultdict
import gym_pacman.envs.util as util


###################################################
#              Environment Setup                  #
###################################################
env = gym.make('BerkeleyPacmanPO-v0')
env.seed(1)
done = False

# episodes, steps, and rewards
n_episodes = 20000
n_steps = 100
graph_steps = 150
rewards = []

# dictionary that will allow us to use the action name as the key
q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])

# Default values for learning algorithms
alpha = 0.05    # smaller learning rates are better, more accurate over time
gamma = 0.9     # high values give bigger weight to rewards
epsilon = 0.3   # high epsilon values = more randomness


###################################################
#              Save Weighted Graph                #
###################################################
def moving_avg_graph(title, file_name):
    # calculate weights
    weights = np.repeat(1.0, graph_steps)/graph_steps
    moving_avg = np.convolve(rewards, weights, 'valid')
    equalized_len = n_episodes - len(moving_avg)

    # get the x and y points
    x = np.arange(equalized_len, n_episodes)
    y = moving_avg

    # scatter plot
    plot.scatter(x, y, marker='.')

    # axis labels and title
    plot.xlabel('Episodes')
    plot.ylabel('Reward')
    plot.title(title)

    # save figure as png
    plot.savefig("plots_and_data/"+file_name)


###################################################
#               E Greedy                          #
###################################################
def policy(s):
    if np.random.uniform(0, 1) < epsilon:
        new_action = env.action_space.sample()
    else:
        new_action = np.argmax(q_table[s])

    return new_action


###################################################
#                  SARSA                          #
###################################################
def learn(s, s_prime, r, a, a_prime):
    current = q_table[s][a]
    estimate = r + gamma*q_table[s_prime][a_prime]
    q_table[s][a] = current + alpha * (estimate - current)


###################################################
#                   Main                          #
###################################################
if __name__ == '__main__':

    for episode in range(n_episodes):
        state = env.reset("trappedClassic.lay")
        action = policy(state)

        for i in range(n_steps):
            # env.render()

            state_prime, reward, done, info = env.step(action)
            action_prime = policy(state_prime)
            learn(state, state_prime, reward, action, action_prime)

            state = state_prime
            action = action_prime

            if done:
                break

        rewards.append(info['episode']['r'])
        print([str(episode), str(info['episode']['r'])])

    moving_avg_graph(str(n_episodes)+'K SARSA',
                     str(n_episodes)+'_sarsa.svg')
    env.close()
