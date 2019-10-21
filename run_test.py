from PIL import Image
import gym
import csv
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


# dictionary that will allow us to use the action name as the key
q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])


# Default values for learning algorithms
alpha = 0.05    # smaller learning rates are better, more accurate over time
gamma = 0.9     # high values give bigger weight to rewards
epsilon = 0.3   # high epsilon values = more randomness


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
#                     Main                        #
###################################################
if __name__ == '__main__':
    # TODO: based on this algorithm http://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html
    episodes = []
    rewards = []
    for j in range(5000):  # 10000 episodes
        gameState = env.reset("smallClassic.lay")

        for i in range(100):    # 100 steps (or until pacman dies or wins)
            # get action
            if util.flipCoin(epsilon):
                #TODO: this doesnt work with state as a deafult dict
                action = np.argmax(q_table[gameState])
                # print(action)
            else:
                action = env.action_space.sample()

            # save old state and features
            old_state = gameState
            # old_features = list(state.values())

            # s <-- s'
            gameState, r, done, info = env.step(action)
            # update_states()

            max_action = max(q_table[gameState])
            q_table[old_state][action] += alpha*((r + gamma*max_action) - q_table[old_state][action])
            # env.render()
            if done:
                break
        # save to csv file for graph
        episodes.append(j)
        rewards.append(info['episode']['r'])
        print([str(j), str(info['episode']['r'])])
    # moving average
    weights = np.repeat(1.0, 100)/100
    moving_avg = np.convolve(rewards, weights, 'valid')
    equalized_len = len(episodes) - len(moving_avg)
    x = episodes[equalized_len:]
    y = moving_avg
    # scatter plots
    plot.scatter(x, y, marker='*')
    # line of best fit
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plot.plot(x, p(x), 'm-')
    # axis labels and title
    plot.xlabel('Episodes')
    plot.ylabel('Reward')
    plot.title('5k Runs')
    # save figure as png
    plot.savefig("5k_run.png")
    env.close()


