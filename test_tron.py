from TRON import tron
from TRON.tron import ConsoleRender
import numpy as np



env = tron.Tron()
crender = ConsoleRender()

if __name__ == "__main__":
    env.reset()

    print(env.env_observation_space)
    print(env.action_space)

    done = False
    choices = env.action_space.vals
    np.random.seed(0)
    while not done:
        c1 = np.random.choice(4, 1, p=[0.1, 0.4, 0.5, 0.0])[0]
        c2 = np.random.choice(4, 1, p=[0.4, 0.1, 0.0, 0.5])[0]
        states, rewards, done = env.step(choices[c1], choices[c2])
        crender.render(env)
        print(rewards)
        print(done)
    