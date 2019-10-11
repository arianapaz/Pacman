import numpy as np
from collections import defaultdict

if __name__ == '__main__':
    state = defaultdict()
    state["Pacman"] = {"x": 1, "y": 2, "Direction": "NORTH"}
    state["Ghost"] = defaultdict(lambda: {"x": 0, "y": 0})
    state["Ghost"][0] = {"x": 1, "y": 2}
    state["Pellets"] = defaultdict(lambda: {"x": 0, "y": 0})
    state["Pellets"][0] = {"x": 1, "y": 2}

    print(state["Ghost"][0])
    print(state["Ghost"][1])
    print(state["Pellets"][0])
    print(state["Pellets"][1])

