from PIL import Image
import gym
import gym_pacman
import time
import numpy as np
from collections import defaultdict
import gym_pacman.envs.util as util

###################################################
#              Environment Setup                  #
###################################################
env = gym.make('BerkeleyPacmanPO-v0')
env.seed(1)
env.chooseLayout(False, "mediumClassic.lay", False)
done = False


###################################################
#                Features Setup                   #
###################################################
features = {'intercept': 1,     # y intercept for the linear equation
            'food_left': 0,
            'pacman_x': 0,
            'pacman_y': 0,
            'ghost_near': 0}    # near(1), not near(0)

# creates a vector the same length as states with random numbers between 1 and 100
weight_vector = np.random.uniform(1, 10, len(features))

# basically creates a 2d dictionary with everything defaulted as 0.0
# the second key is guaranteed to be one of the actions so i just did that instead
q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])

# Default values for learning algorithms
alpha = 0.05    # smaller learning rates are better, more accurate over time
gamma = 0.9     # high values give bigger weight to rewards
epsilon = 0.7   # high epsilon values = more randomness


def update_features():
    features['food_left'] = info['num_food']
    features['pacman_x'] = info['curr_loc'][0]
    features['pacman_y'] = info['curr_loc'][1]
    # states['ghost_positions'] = info['ghost_positions']
    features['ghost_near'] = int(info['ghost_in_frame'])


# TODO: i used this two websites as main references
#  https://frnsys.com/ai_notes/artificial_intelligence/reinforcement_learning.html
#  https://www.cs.swarthmore.edu/~bryce/cs63/s18/slides/3-23_approximate_ql.pdf
#  _
#  The second reference gives a formula that uses V(s') to calculate the correction
#  but the first one uses max of Q(s',a') but since idk if we will have that I left it as V(s')
#  _
#  I finally figured out that the last part of the SGD algorithm is
#  the feature value of s,a (some refer to it as f_i or f(s,a) or theta(s))
#  You can double check this in both links and even on amy's notes.
def update_weights(s, a, reward, s_prime, feat):

    # TODO: create funtion feat(s,a) that returns a feature given s,a
    for weight in range(len(weight_vector)):
        max_q = max(q_table[s_prime])
        correction = (reward + gamma*max_q) - q_table[s][a]
        theta = feat[weight]
        weight_vector[weight] = weight_vector[weight] + alpha*correction*theta


if __name__ == '__main__':
    # TODO: based on this algorithm
    #  http://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html
    for j in range(10000):  # 10000 episodes
        state = env.reset("mediumClassic.lay")
        for i in range(100):    # 100 steps (or until pacman dies or wins)
            if util.flipCoin(epsilon):
                action = np.argmax(q_table[state])
            else:
                action = env.action_space.sample()
            old_state = state
            old_features = list(features.values())
            # s <-- s'
            state, r, done, info = env.step(action)
            update_features()
            w_t = np.transpose(weight_vector)
            # update Q(s,a)
            q_table[old_state][action] = float(np.dot(w_t, list(features.values())))
            # update the weights
            update_weights(old_state, action, r, state, old_features)
            env.render()
            if done:
                break
        print(info['episode']['r'])
    env.close()


