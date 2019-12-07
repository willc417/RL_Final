import numpy as np
import gym
from retrace_lambda import RetraceLambda, StateActionFeatureVectorWithTile
import pandas as pd

def test_sarsa_lambda():


    env = gym.make("MountainCar-v0") #gym.make("MountainCar-v0")
    gamma = 1.

    X = StateActionFeatureVectorWithTile(
        env.observation_space.low, #np.array([-3, -3.5, -0.25, -3.5])
        env.observation_space.high, #np.array([3, 3.5, 0.25, 3.5]),
        env.action_space.n,
        num_tilings=10,
        tile_width=np.array([.45, .035]) #[.45, 50, .45, 50]) #Need to adjust the tile width for CartPole
    )


    def greedy_policy(s,done):
        Q = [np.dot(w, X(s,done,a)) for a in range(env.action_space.n)]
        return np.argmax(Q)

    def _eval(render=False):
        s, done = env.reset(), False
        if render: env.render()

        G = 0.
        while not done:
            a = greedy_policy(s,done)
            s,r,done,_ = env.step(a)
            if render: env.render()

            G += r
        return G
    lam = 1
    lambda_values = []
    return_values = []
    for i in range(0,10):
        w = RetraceLambda(env, gamma, lam, 0.01, X, 10)
        Gs = [_eval() for _ in  range(100)]
        _eval(False)
        lambda_values.append(lam)
        print(np.max(Gs))
        return_values.append(np.max(Gs))
        lam -= 0.1

    sarsa_lambda_data = pd.DataFrame(data={"Lambda Values": lambda_values, "Max Rewards": return_values})
    sarsa_lambda_data.to_csv("sarsa_lambda_returns.csv")
    return sarsa_lambda_data


if __name__ == "__main__":
    test_sarsa_lambda()
