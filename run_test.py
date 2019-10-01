from PIL import Image
import gym
import gym_pacman
import time
import numpy as np
from collections import defaultdict
# TODO: I'm going push this in case you look at it before i do tomorrow. I'm mainly just confused about the order of
#   of things right now because its like 4 am. I have everything to update the weights, but i still don't know how to
#   do the gradient part. I'm sure you'll have questions about my haphazard code, but if you look at it in the morning
#   try to do the update weights methods and just assume you have everything you need

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
weight_vector = np.random.randint(1, 10, len(features))

# basically creates a 2d dictionary with everything defaulted as 0.0,
# things are only created when they're indexed
q_table = defaultdict(lambda: defaultdict(float))

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
def update_weights(s, a, r, s_prime):

    # TODO: create funtion feat(s,a) that returns a feature given s,a
    # TODO: create a function getValue(s) given s (or s')
    for weight in range(len(weight_vector)):
        correction = (r + gamma*getValue(s_prime)) - q_table[s][a]
        theta = feat(s,a)
        weight_vector[weight] = weight_vector[weight] + alpha*correction*theta


if __name__ == '__main__':

    init_state = env.reset("mediumClassic.lay")
    action = env.action_space.sample()

    # TODO: based on this algorithm
    #  http://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html
    for j in range(10000):  # 10000 episodes
        state, r, done, info = env.step(action)

        for i in range(100):    # 100 steps (or until pacman dies or wins)
            # TODO: choose a from s using policy derived from Q
            # TODO: take action a, observe r,s'
            # TODO: update Q(s,a)
            # TODO: s <-- s'
            # TODO: repeat this process until s is teminal

            
            # TODO: to generate next state(s_prime) do state.generatePacmanSuccessor(action)
            update_features()
            q_table[state][action] = float(np.dot(w_t, list(features.values())))
            # TODO: update the weights
            update_weights(state,action,r,s_prime)
            # TODO: with probability epsilon take
            #   action = argmax a of Q(s, a). argmax returns the argument a for which Q(s,a) is the largest
            #   otherwise take a random action
            action = env.action_space.sample()
            env.render()
            if done:
                break
        print(info['episode'])
    env.close()


