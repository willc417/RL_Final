import numpy as np
import gym
from retrace_lambda import RetraceLambda
from StateActionFeatureVector import StateActionFeatureVectorWithTile
import argparse
import time
import pandas as pd


def test_retrace_lambda(num_episodes = None):

    if num_episodes is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--num_eps', default=10, type=int)
        args = parser.parse_args()
        num_episodes = args.num_eps

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

    max_values = []
    min_values = []

    r_lambda_rpe = pd.DataFrame()

    for i in range(0,5):
        start_time = time.time()
        w, reward_list = RetraceLambda(env, gamma, lam, 0.01, X, num_episodes)
        total_time = time.time() - start_time
        print("Retrace Lambda (Lambda = {}) training time with {} episodes: time: {} s".format(round(lam, 2), num_episodes, total_time))
        Gs = [_eval() for _ in  range(100)]
        _eval(False)
        print(np.max(Gs))
        return_values.append(np.max(Gs))
        max_values.append(np.max(Gs))
        min_values.append(np.min(Gs))
        r_lambda_rpe.insert(i, str(round(lam, 2)), reward_list)
        lam -= 0.2

    r_eps_data = pd.DataFrame(data={"Lambda Values": lambda_values, "Max Rewards": max_values, "Min Rewards": min_values})
    r_eps_data.to_csv('retrace_lambda_returns.csv', index=False)
    
    #sarsa_lambda_data = pd.DataFrame(data={"Lambda Values": lambda_values, "Max Rewards": max_values, "Min Rewards": min_values})
    r_lambda_rpe.to_csv("retrace_lambda_rewards_per_episode.csv", index=False)
    return r_eps_data, r_lambda_rpe


if __name__ == "__main__":
    test_retrace_lambda()
