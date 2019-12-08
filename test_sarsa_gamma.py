from StateActionFeatureVector import StateActionFeatureVectorWithTile
from sarsa_gamma import sarsa_gamma
import numpy as np
import pandas as pd
import argparse
import gym
import time

def test_sarsa_gamma(num_episodes=None):

    env = gym.make("MountainCar-v0")
    if num_episodes is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--num_eps', default=10, type=int)
        args = parser.parse_args()
        num_episodes = args.num_eps

    gamma_values = []
    max_values = []
    min_values = []
    gamma = 1.
    start_time = time.time()
    w, rewards_per_episode = sarsa_gamma(num_episodes, gamma)
    total_time = time.time() - start_time
    print("Sarsa Gamma training time with {} episodes: {} s".format(num_episodes, total_time))
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
    max_values.append(np.max(Gs))
    min_values.append(np.min(Gs))
        #assert np.max(Gs) >= -110.0, 'fail to solve mountaincar'
    sarsa_gamma_data = pd.DataFrame(data={"Gamma Values": gamma_values, "Max Rewards": max_values, "Min Rewards": min_values})
    sarsa_gamma_data.to_csv("sarsa_gamma_returns.csv", index=False)
    sarsa_gamma_rewards_per_episode = pd.DataFrame(data={"Gamma": rewards_per_episode})
    sarsa_gamma_rewards_per_episode.to_csv("sarsa_gamma_rpe.csv", index=False)
    return sarsa_gamma_data, sarsa_gamma_rewards_per_episode

if __name__ == "__main__":
    test_sarsa_gamma()