import numpy as np
import gym
from retrace_lambda import RetraceLambda
from StateActionFeatureVector import StateActionFeatureVectorWithTile
import argparse

import pandas as pd

def test_sarsa_lambda():
    
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
    #num_episodes = 100
    ovr_reward_list = []

    lambda_ovr= []

    count_eps = int(num_episodes/10)
    episode_list = [i+1 for i in range(int(num_episodes/10))]
    max_values = []
    min_values = []

    r_lambda_rpe = pd.DataFrame()

    for i in range(0,5):
        w, reward_list = RetraceLambda(env, gamma, lam, 0.01, X, num_episodes)
        Gs = [_eval() for _ in  range(100)]
        _eval(False)
        print(np.max(Gs))
        lambda_values.append(lam)
        return_values.append(np.max(Gs))
        ovr_reward_list.append(reward_list)
        max_values.append(np.max(Gs))
        min_values.append(np.min(Gs))
        lambda_ovr.append([lam] * len(reward_list))
        r_lambda_rpe.insert(i, str(round(lam, 2)), reward_list)
        lam -= 0.2

    ovr_reward_list = [item for sublist in ovr_reward_list for item in sublist]
    lambda_ovr = [item for sublist in lambda_ovr for item in sublist]

    r_eps_data = pd.DataFrame(data={"Lambda Values": lambda_values, "Max Rewards": max_values, "Min Rewards": min_values})
    r_eps_data.to_csv('retrace_lambda_returns.csv', index=False)
    
    #sarsa_lambda_data = pd.DataFrame(data={"Lambda Values": lambda_values, "Max Rewards": max_values, "Min Rewards": min_values})
    r_lambda_rpe.to_csv("retrace_lambda_rewards_per_episode.csv", index=False)
    return r_eps_data, r_lambda_rpe


if __name__ == "__main__":
    test_sarsa_lambda()
