import gym
import numpy as np
import matplotlib.pyplot as plot
from collections import defaultdict
import gym_pacman.envs.util as util

# TODO: based on this algorithm http://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html


###################################################
#              Environment Setup                  #
###################################################
env = gym.make('BerkeleyPacmanPO-v0')
env.seed(1)
done = False

# episodes, steps, and rewards
n_episodes = 5000
n_steps = 100
rewards = []

# dictionary that will allow us to use the action name as the key
q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])

# Default values for learning algorithms
alpha = 0.05    # smaller learning rates are better, more accurate over time
gamma = 0.7     # high values give bigger weight to rewards
epsilon = 0.3   # high epsilon values = more randomness


###################################################
#        State and Learning Rate Setup            #
###################################################
state = defaultdict()
state["pacman"] = {"x": 0, "y": 0, "dir": "NORTH"}
state["ghost"] = defaultdict(lambda: {"x": 0, "y": 0, "dir": 0})
state["pellets"] = defaultdict(lambda: {"x": 0, "y": 0})
state["ghost_edible"] = False
state["num_edible_ghosts"] = 0
state["pellets_left"] = 0
state["power_pellets_timer"] = 0
state["power_pellets_left"] = 0


# TODO: need to fix this
###################################################
#                 Update States                   #
###################################################
def update_states():
    # update pacman's info
    state['pacman'] = {"x": info['curr_loc'][0],
                       "y": info['curr_loc'][1],
                       "dir": info['curr_orientation']}

    # update the ghost's positions
    for num in range(len(info['ghost_positions'])):
        state['ghost'][num] = {"x": info['ghost_positions'][num][0],
                               "y": info['ghost_positions'][num][1]}

    # update info about the pellets
    state['pellets_left'] = info['num_food']


###################################################
#              Save Weighted Graph                #
###################################################
def moving_avg_graph(title, file_name):
    # calculate weights
    weights = np.repeat(1.0, 100)/100
    moving_avg = np.convolve(rewards, weights, 'valid')
    equalized_len = n_episodes - len(moving_avg)

    # get the x and y points
    x = np.arange(equalized_len, n_episodes)
    y = moving_avg

    # scatter plot
    plot.scatter(x, y, marker='.')

    # line of best fit
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plot.plot(x, p(x), 'm-')

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
#                Q-learning                       #
###################################################
def learn(s, s_prime, r, a):
    # TODO: this doesnt work with state as a default dict
    max_action = max(q_table[s_prime])
    current = q_table[s][a]
    estimate = r + gamma * max_action
    q_table[s][a] = current + alpha * (estimate - current)


###################################################
#                     Main                        #
###################################################
if __name__ == '__main__':
    for episode in range(n_episodes):
        state = env.reset("smallClassic.lay")
        action = policy(state)

        for i in range(n_steps):
            # env.render()
            state_prime, reward, done, info = env.step(action)
            learn(state, state_prime, reward, action)

            state = state_prime

            if done:
                break

        rewards.append(info['episode']['r'])
        print([str(episode), str(info['episode']['r'])])

    moving_avg_graph(str(n_episodes)+'K Q-learning',
                     str(n_episodes)+'K_q_learning')
    env.close()
