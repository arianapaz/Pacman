from PIL import Image
import gym
import gym_pacman
import time

env = gym.make('BerkeleyPacmanPO-v0')
env.seed(1)
env.chooseLayout(False, "originalClassic.lay", False)
done = False
env.reset("originalClassic.lay")
i = 0
for i in range(100):
    i += 1
    s_, r, done, info = env.step(env.action_space.sample())
    print(info)
    env.render()
env.close()
