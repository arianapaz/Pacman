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

env = gym.make('BerkeleyPacmanPO-v0')
env.seed(1)
env.chooseLayout(False, "mediumClassic.lay", False)
done = False
# ghost near is 0 or 1, intercept is just the y intercept of the linear equation
features = {'intercept': 1, 'food_left': 0, 'pacman_x': 0, 'pacman_y': 0, 'ghost_near': 0}
# creates a vector the same length as states with random numbers between 1 and 100
weight_vector = np.random.randint(1, 10, len(features))
# basically creates a 2d dictionary with everything defaulted as 0.0, things are only created when they're indexed
q_table = defaultdict(lambda: defaultdict(float))
# TODO: are these good values?
alpha = 0.5
gamma = 0.9
epsilon = 0.7


def update_features():
    features['food_left'] = info['num_food']
    features['pacman_x'] = info['curr_loc'][0]
    features['pacman_y'] = info['curr_loc'][1]
    # states['ghost_positions'] = info['ghost_positions']
    features['ghost_near'] = int(info['ghost_in_frame'])


def update_weights(s, a, r, s_prime):
    # TODO: you can access the q table like this: q_table[s][a]
    pass


if __name__ == '__main__':
    # 10000 episodes
    for j in range(10000):
        init_state = env.reset("mediumClassic.lay")
        # 100 steps in an episode, or until pacman dies or wins
        action = env.action_space.sample()
        for i in range(100):
            # TODO: do we calculate q before or after taking an action?
            w_t = np.transpose(weight_vector)
            state, r, done, info = env.step(action)
            # TODO: to generate next state(s_prime) do state.generatePacmanSuccessor(action)
            update_features()
            q_table[state][action] = float(np.dot(w_t, list(features.values())))
            # TODO: update the weights
            # TODO: with probability epsilon take
            #   action = argmax a of Q(s, a). argmax returns the argument a for which Q(s,a) is the largest
            #   otherwise take a random action
            action = env.action_space.sample()
            env.render()
            if done:
                break
        print(info['episode'])
    env.close()


