import numpy as np
from collections import defaultdict

if __name__ == '__main__':
    q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    q_table["state"][0] = 1.2
    q_table["state"][1] = 7.6
    q_table["state"][3] = 4.5
    q_table["state1"][0] = 11.9
    q_table["state1"][1] = 6.8
    q_table["state1"][3] = 8.3
    q_table["state1"][4] = 28.7
    action_of_state = q_table["state1"]
    print(action_of_state)
    print(max(action_of_state))
    print(np.argmax(action_of_state))
