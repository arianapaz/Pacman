import pickle
import numpy as np
from collections import defaultdict

if __name__ == '__main__':
    # state = defaultdict()
    # state["Pacman"] = {"x": 1, "y": 2, "Direction": "NORTH"}
    # state["Ghost"] = defaultdict(lambda: {"x": 0, "y": 0})
    # state["Ghost"][0] = {"x": 1, "y": 2}
    # state["Pellets"] = defaultdict(lambda: {"x": 0, "y": 0})
    # state["Pellets"][0] = {"x": 1, "y": 2}
    #
    # print(state["Ghost"][0])
    # print(state["Ghost"][1])
    # print(state["Pellets"][0])
    # print(state["Pellets"][1])
    # save q table

    q_table = {'a': 0.0, 'e': 0.0, 'i': 0.0, 'o': 0.0, 'u': 0.0}

    f = open('plots_and_data/q_table/q_table.pkl', 'wb')
    pickle.dump(q_table, f)
    f.close()

    with open('plots_and_data/q_table/q_table.pkl', 'rb') as f:
        x = pickle.load(f)
    print(x)
