from Feature_Representation.StateActionFeatureVector import StateActionFeatureVectorWithTile
from Retrace.retrace_gamma import retrace_gamma
import numpy as np
import gym
import pandas as pd
import time
import argparse


def test_retrace_gamma(num_episodes=None):
    env = gym.make("MountainCar-v0")

    gamma_values = []
    max_values = []
    min_values = []

    return_values = []
    if num_episodes is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--num_eps', default=10, type=int)
        args = parser.parse_args()
        num_episodes = args.num_eps

    gamma = 1
    start_time = time.time()
    w, reward_list = retrace_gamma(num_episodes, gamma)
    total_time = time.time() - start_time
    print("Retrace Gamma training time with {} episodes: {} s".format(num_episodes, total_time))
    X = StateActionFeatureVectorWithTile(
        env.observation_space.low,
        env.observation_space.high,
        env.action_space.n,
        num_tilings=10,
        tile_width=np.array([.45, .035])
    )

    num_episodes_list = [i+1 for i in range(num_episodes)]
    #print(reward_list)

    def greedy_policy(s, done):
        Q = [np.dot(w, X(s, done, a)) for a in range(env.action_space.n)]
        return np.argmax(Q)

    def _eval(render=False):
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

    gamma_values.append(gamma)
    return_values.append(np.max(Gs))
    max_values.append(np.max(Gs))
    min_values.append(np.min(Gs))
    # assert np.max(Gs) >= -110.0, 'fail to solve mountaincar'
    episodes_data = pd.DataFrame(data={"Gamma Values": gamma_values, "Max Rewards": max_values, "Min Rewards": min_values})
    episodes_data.to_csv('retrace_gamma_returns.csv', index=False)

    retrace_gamma_rewards_per_episode = pd.DataFrame(data={"Gamma": reward_list})
    retrace_gamma_rewards_per_episode.to_csv("retrace_gamma_rpe.csv", index=False)
    return episodes_data, retrace_gamma_rewards_per_episode

if __name__ == "__main__":
    test_retrace_gamma()
