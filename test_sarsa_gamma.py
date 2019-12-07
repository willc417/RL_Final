from StateActionFeatureVector import StateActionFeatureVectorWithTile
from sarsa_gamma import sarsa_gamma
import numpy as np
import pandas as pd
import gym

def test_sarsa_gamma():

    env = gym.make("MountainCar-v0")

    gamma_values = []
    return_values = []
    gamma = 1.

    w = sarsa_gamma(10, gamma)
    X = StateActionFeatureVectorWithTile(
        env.observation_space.low,
        env.observation_space.high,
        env.action_space.n,
        num_tilings=10,
        tile_width=np.array([.45, .035])
    )

    def greedy_policy(s, done):
        Q = [np.dot(w, X(s, done, a)) for a in range(env.action_space.n)]
        return np.argmax(Q)

    def _eval(render=False):
        # print('hello')
        s, done = env.reset(), False
        if render: env.render()

        G = 0.
        while not done:
            a = greedy_policy(s, done)
            s, r, done, _ = env.step(a)
            if render: env.render()

            G += r
        return G

    Gs = [_eval() for _ in range(100)]
    _eval(False)

    print(np.max(Gs))
    gamma_values.append(gamma)
    return_values.append(np.max(Gs))
        #assert np.max(Gs) >= -110.0, 'fail to solve mountaincar'
    sarsa_gamma_data = pd.DataFrame(data={"Gamma Values": gamma_values, "Max Rewards": return_values})
    sarsa_gamma_data.to_csv("sarsa_gamma_returns.csv")
    return sarsa_gamma_data

test_sarsa_gamma()