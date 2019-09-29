from PIL import Image
import gym
import gym_pacman
import time

env = gym.make('BerkeleyPacmanPO-v0')
env.seed(1)
done = False
env.chooseLayout(False, "originalClassic.lay", False)
env.reset("originalClassic.lay")
for i in range(1000):
    done = False
    s_, r, done, info = env.step(env.action_space.sample())
    env.render()
env.close()