from PIL import Image
import gym
import gym_pacman
import time
import numpy as np

def update_states():
    states['food_left'] = info['num_food']
    states['pacman_x'] = info['curr_loc'][0]
    states['pacman_y'] = info['curr_loc'][1]
    # states['ghost_positions'] = info['ghost_positions']
    states['ghost_near'] = info['ghost_in_frame']


env = gym.make('BerkeleyPacmanPO-v0')
env.seed(1)
env.chooseLayout(False, "mediumClassic.lay", False)
done = False
states = {'food_left': 0, 'pacman_x': 0, 'pacman_y': 0, 'ghost_near': False}
q_table = np.zeros((env.action_space.n, len(states)))
# 1 episode, for more episode loop over this
env.reset("mediumClassic.lay")
i = 0
# 100 steps in an episode
for i in range(100):
    i += 1
    s_, r, done, info = env.step(env.action_space.sample())
    update_states()
    print(states)
    env.render()
    if done:
        break
print(info['episode'])
env.close()
