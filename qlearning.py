import gym
import numpy as np
import matplotlib.pyplot as plot
from collections import defaultdict
import gym_pacman.envs.util as util

# based on this algorithm http://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html


###################################################
#              Environment Setup                  #
###################################################
env = gym.make('BerkeleyPacmanPO-v0')
env.seed(1)
done = False

# episodes, steps, and rewards
n_episodes = 2000
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
# 1-4: iff taking this action will cause an illegal action {0,1}
# 5-8: if there is a ghost within 8 spaces following us in this direction {0, 1}
# 9: direction to nearest pellet/power pellet/scared ghost {0, len(map_height) || len(map_width)}
# 10: If we cannot move in any direction without dying {0, 1}
state = {"ill_n": 0, "ill_e": 0, "ill_s": 0, "ill_w": 0,
         "n_ghost": 0, "e_ghost": 0, "s_ghost": 0, "w_ghost": 0,
         "dir_pellet": 0, "trapped": 0}


# TODO: need to fix this
###################################################
#                 Update States                   #
###################################################
def update_states(gameInfo):
    pass


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
    plot.savefig("plots_and_data/"+file_name, dpi=1200)


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

        for i in range(n_steps):
            # env.render()
            
            action = policy(state)
            state_prime, reward, done, info = env.step(action)
            learn(state, state_prime, reward, action)
            state = state_prime

            if done:
                break

        rewards.append(info['episode']['r'])
        print([str(episode), str(info['episode']['r'])])

    moving_avg_graph(str(n_episodes)+'K Q-learning',
                     str(n_episodes)+'K_q_learning.svg')
    env.close()
