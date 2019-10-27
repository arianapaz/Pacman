import gym
import numpy as np
import pickle
import matplotlib.pyplot as plot
from gym_pacman.envs.game import Actions
from collections import defaultdict
import gym_pacman.envs.util as util

###################################################
#              Environment Setup                  #
###################################################
env = gym.make('BerkeleyPacmanPO-v0')
env.seed(1)
done = False

# episodes, steps, and rewards
n_episodes = 500
n_steps = 100
graph_steps = 150
rewards = []

# dictionary that will allow us to use the action name as the key
q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
# q_table = {'North': 0, 'South': 0, 'East': 0, 'West': 0, 'Stop': 0}


# Default values for learning algorithms
alpha = 0.05  # smaller learning rates are better, more accurate over time
gamma = 0.9  # high values give bigger weight to rewards
epsilon = 0.3  # high epsilon values = more randomness

###################################################
#        State and Learning Rate Setup            #
###################################################
# 1-4: iff taking this action will cause an illegal action {0,1}
# 5-8: if there is a ghost within 8 spaces following us in this direction {0, 1}
# 9: direction of nearest pellet/power pellet/scared ghost {0, len(map_height) || len(map_width)}
# 10: If we cannot move in any direction without dying {0, 1}
initial_state = {"illegal_north": 0, "illegal_east": 0, "illegal_south": 0, "illegal_west": 0,
                 "ghost_north": 0, "ghost_east": 0, "ghost_south": 0, "ghost_west": 0,
                 "nearest_pellet": 0, "trapped": 0}
# weights = {"illegal_north": 0, "illegal_east": 0, "illegal_south": 0, "illegal_west": 0,
#            "ghost_north": 0, "ghost_east": 0, "ghost_south": 0, "ghost_west": 0,
#            "nearest_pellet": 0, "trapped": 0}


# TODO: currently working on this
###################################################
#                 Update States                   #
###################################################
def update_states(copy_state, game_info):
    # info needed
    pacman = [float(game_info['curr_loc'][0]), float(game_info['curr_loc'][1])]
    food = game_info['food_location']
    walls = game_info['wall_positions']
    capsules = game_info['power_pellets_locations']
    ghost_positions = game_info['ghost_positions']
    scared_timer = game_info['scared_timer']
    obstacle = {"north": False, "east": False, "south": False, "west": False}

    # update the illegal actions
    copy_state['illegal_north'] = int('North' not in game_info['legal_actions'])
    copy_state['illegal_east'] = int('East' not in game_info['legal_actions'])
    copy_state['illegal_south'] = int('South' not in game_info['legal_actions'])
    copy_state['illegal_west'] = int('West' not in game_info['legal_actions'])

    # update the ghost threats
    for ghost in ghost_positions:
        if abs(pacman[0] - ghost[0]) <= 4 or abs(pacman[1] - ghost[1]) <= 4:
            if pacman[0] == ghost[0]:
                if no_walls(walls, int(pacman[0]), pacman[1], ghost[1], True):
                    copy_state['ghost_north'] = int(pacman[1] < ghost[1])
                    copy_state['ghost_south'] = int(pacman[1] > ghost[1])
                    copy_state['ghost_east'] = 0
                    copy_state['ghost_west'] = 0
                else:
                    copy_state['ghost_north'] = 0
                    copy_state['ghost_east'] = 0
                    copy_state['ghost_south'] = 0
                    copy_state['ghost_west'] = 0
            elif pacman[1] == ghost[1]:
                if no_walls(walls, int(pacman[1]), pacman[0], ghost[0], False):
                    copy_state['ghost_east'] = int(pacman[0] < ghost[0])
                    copy_state['ghost_west'] = int(pacman[0] > ghost[0])
                    copy_state['ghost_north'] = 0
                    copy_state['ghost_south'] = 0
                else:
                    copy_state['ghost_north'] = 0
                    copy_state['ghost_east'] = 0
                    copy_state['ghost_south'] = 0
                    copy_state['ghost_west'] = 0
            else:
                copy_state['ghost_north'] = 0
                copy_state['ghost_east'] = 0
                copy_state['ghost_south'] = 0
                copy_state['ghost_west'] = 0
        else:
            copy_state['ghost_north'] = 0
            copy_state['ghost_east'] = 0
            copy_state['ghost_south'] = 0
            copy_state['ghost_west'] = 0

    # update the closest food/scared ghost/capsule
    if scared_timer > 10:
        copy_state['nearest_pellet'] = closest_food(pacman, food, walls, capsules, ghost_positions)
    else:
        copy_state['nearest_pellet'] = closest_food(pacman, food, walls, capsules)

    # update trapped situation
    obstacle["north"] = ((copy_state["illegal_north"] or copy_state["ghost_north"]) > 0)
    obstacle["south"] = ((copy_state["illegal_south"] or copy_state["ghost_south"]) > 0)
    obstacle["east"] = ((copy_state["illegal_east"] or copy_state["ghost_east"]) > 0)
    obstacle["west"] = ((copy_state["illegal_west"] or copy_state["ghost_west"]) > 0)

    if obstacle["south"] and obstacle["north"] and obstacle["west"] and obstacle["east"]:
        copy_state['trapped'] = 1
    else:
        copy_state['trapped'] = 0
    return copy_state


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
        for col in np.arange(int(low), int(high) + 1):
            if wall[index][col]:
                return False
        return True

    else:
        for row in np.arange(int(low), int(high) + 1):
            if wall[row][index]:
                return False
        return True


###################################################
#       Nearest Food/Capsules/Ghost               #
###################################################
def closest_food(pacman, food, walls, capsules, ghost_locations=None):
    fringe = [(int(pacman[0]), int(pacman[1]), 0)]
    expanded = set()
    direction_found = 4

    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        for capsule in capsules:
            if float(capsule[0]) == pos_x and float(capsule[1]) == pos_y:
                if int(pacman[0]) == pos_x:
                    if int(pacman[1]) < pos_y:
                        direction_found = 0
                    else:
                        direction_found = 2
                elif int(pacman[1]) == pos_y:
                    if int(pacman[0]) < pos_x:
                        direction_found = 1
                    else:
                        direction_found = 3
                break
        if ghost_locations is not None:
            for ghost in ghost_locations:
                if ghost[0] == pos_x and ghost[1] == pos_y:
                    if float(ghost[0]) == pos_x and float(ghost[1]) == pos_y:
                        if int(pacman[0]) == pos_x:
                            if int(pacman[1]) < pos_y:
                                direction_found = 0
                            else:
                                direction_found = 2
                        elif int(pacman[1]) == pos_y:
                            if int(pacman[0]) < pos_x:
                                direction_found = 1
                            else:
                                direction_found = 3
                    break
        if food[pos_x][pos_y]:
            if int(pacman[0]) == pos_x:
                if int(pacman[1]) < pos_y:
                    direction_found = 0
                else:
                    direction_found = 2
            elif int(pacman[1]) == pos_y:
                if int(pacman[0]) < pos_x:
                    direction_found = 1
                else:
                    direction_found = 3
            break
        # otherwise spread out from the location to its neighbours
        neighbors = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in neighbors:
            fringe.append((nbr_x, nbr_y, dist + 1))

    return direction_found


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
#          Approximate Q-learning                 #
###################################################
# def approximate_learn(s, s_key, s_prime_key, r, a):
#     max_action = max(q_table[s_prime_key])
#     current = q_table[s_key][a]
#     correction = (r + gamma * max_action) - current
#     weighted_sum = 0
#     for feature in weights:
#         weights[feature] += alpha * correction * s[feature]
#         weighted_sum += weights[feature] * s[feature]
#     q_table[s_key][a] = weighted_sum


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
    plot.savefig("plots_and_data/" + file_name, dpi=1200)


###################################################
#                     Main                        #
###################################################
if __name__ == '__main__':
    for episode in range(n_episodes):
        # env.reset("trappedClassic.lay")
        state = env.reset("trappedClassic.lay")

        # getting initial state and its dictionary key value
        # state = initial_state
        # state_key = int("".join(map(str, state.values())))

        for i in range(n_steps):
            # env.render()

            action = policy(state)
            # action = policy(state_key)
            # _, reward, done, info = env.step(action)
            state_prime, reward, done, info = env.step(action)

            # state_prime = update_states(state, info)
            # state_prime_key = int("".join(map(str, state_prime.values())))

            learn(state, state_prime, reward, action)
            # learn(state_key, state_prime_key, reward, action)
            # approximate_learn(state, state_key, state_prime_key, reward, action)

            state = state_prime
            # state_key = state_prime_key

            if done:
                break

        rewards.append(info['episode']['r'])
        print([str(episode), str(info['episode']['r'])])

    # save graph
    moving_avg_graph(str(n_episodes) + ' Q-learning',
                     str(n_episodes) + '_q_learning.svg')

    # save q table
    f = open('plots_and_data/q_table/q_table.pkl', 'wb')
    pickle.dump(q_table, f)
    f.close()

    with open('plots_and_data/q_table/q_table.pkl', 'rb') as f:
        x = pickle.load(f)
    print(x)

    env.close()
