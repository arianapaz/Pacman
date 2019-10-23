import gym
import numpy as np
from gym_pacman.envs.game import Actions
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
n_episodes = 10000
n_steps = 100
rewards = []

# dictionary that will allow us to use the action name as the key
q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])

# Default values for learning algorithms
alpha = 0.05    # smaller learning rates are better, more accurate over time
gamma = 0.9     # high values give bigger weight to rewards
epsilon = 0.3   # high epsilon values = more randomness


###################################################
#        State and Learning Rate Setup            #
###################################################
# 1-4: iff taking this action will cause an illegal action {0,1}
# 5-8: if there is a ghost within 8 spaces following us in this direction {0, 1}
# 9: direction to nearest pellet/power pellet/scared ghost {0, len(map_height) || len(map_width)}
# 10: If we cannot move in any direction without dying {0, 1}
state = {"illegal_north": 0, "illegal_east": 0, "illegal_south": 0, "illegal_west": 0,
         "ghost_north": 0, "ghost_east": 0, "ghost_south": 0, "ghost_west": 0,
         "nearest_pellet": 0, "trapped": 0}


# TODO: currently working on this
###################################################
#                 Update States                   #
###################################################
def update_states(game_info):
    # info needed
    pacman = [float(game_info['curr_loc'][0]), float(game_info['curr_loc'][1])]
    food = game_info['food_location']
    walls = game_info['wall_positions']
    capsules = game_info['power_pellets_locations']
    ghost_positions = game_info['ghost_positions']
    scared_timer = game_info['scared_timer']

    # update the illegal actions
    state['illegal_north'] = 'North' not in game_info['legal_actions']
    state['illegal_east'] = 'East' not in game_info['legal_actions']
    state['illegal_south'] = 'South' not in game_info['legal_actions']
    state['illegal_west'] = 'West' not in game_info['legal_actions']

    # update the ghost threats
    for ghost in ghost_positions:
        if abs(pacman[0]-ghost[0]) <= 4 or abs(pacman[1]-ghost[1]) <= 4:
            if pacman[0] == ghost[0]:
                if no_walls(walls, pacman[0], pacman[1], ghost[1], True):
                    state['ghost_north'] = pacman[1] < ghost[1]
                    state['ghost_south'] = pacman[1] > ghost[1]
                    state['ghost_east'] = 0
                    state['ghost_west'] = 0
                else:
                    state['ghost_north'] = 0
                    state['ghost_east'] = 0
                    state['ghost_south'] = 0
                    state['ghost_west'] = 0
            elif pacman[1] == ghost[1]:
                if no_walls(walls, pacman[1], pacman[0], ghost[0], False):
                    state['ghost_east'] = pacman[0] < ghost[0]
                    state['ghost_west'] = pacman[0] > ghost[0]
                    state['ghost_north'] = 0
                    state['ghost_south'] = 0
                else:
                    state['ghost_north'] = 0
                    state['ghost_east'] = 0
                    state['ghost_south'] = 0
                    state['ghost_west'] = 0
            else:
                state['ghost_north'] = 0
                state['ghost_east'] = 0
                state['ghost_south'] = 0
                state['ghost_west'] = 0
        else:
            state['ghost_north'] = 0
            state['ghost_east'] = 0
            state['ghost_south'] = 0
            state['ghost_west'] = 0

    # update direction of nearest pellet/power-pellet/eatable-ghost
    # TODO: make a method like "no_walls" that gets the closest food
    # TODO: add if statements to determine the direction
    if scared_timer > 10:
        state['nearest_pellet'] = closest_food(pacman, food, walls, capsules, ghost_positions)
    else:
        state['nearest_pellet'] = closest_food(pacman, food, walls, capsules)

    # update trapped situation
    # TODO: need check if there is a ghost or wall on every side
    if state['illegal_north'] and state['illegal_south'] and \
       state['illegal_east'] and state['illegal_west']:
        state['trapped'] = 1


###################################################
#               No Walls                          #
###################################################
def no_walls(wall, index, bound1, bound2, is_row):
    if bound1 > bound2:
        low = bound2
        high = bound1
    else:
        low = bound1
        high = bound2

    if is_row:
        for col in np.arange(low, high + 1.):
            if wall[index][col]:
                return False
        return True

    else:
        for row in np.arange(low, high + 1.):
            if wall[row][index]:
                return False
        return True


###################################################
#       Nearest Food/Capsules/Ghost               #
###################################################
def closest_food(pos, food, walls, capsules, ghost_locations=None):
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        for capsule in capsules:
            if float(capsule[0]) == pos_x and float(capsule[1]) == pos_y:
                return dist
        if ghost_locations is not None:
            for ghost in ghost_locations:
                if ghost[0] == pos_x and ghost[1] == pos_y:
                    return dist
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        neighbors = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in neighbors:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return -1


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
    max_action = max(q_table[s_prime])
    current = q_table[s][a]
    estimate = r + gamma * max_action
    q_table[s][a] = current + alpha * (estimate - current)


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
