import numpy as np
import hac_envs  # noqa: F401
import gymnasium as gym


env = gym.make("PendulumHAC-v1")
print(env.reset())

print(env.step(np.zeros(1)))


env.close()
