import numpy as np
import gym
import pandas as pd
from time import sleep
from StateActionFeatureVector import StateActionFeatureVectorWithTile

class GenEstimator():
    def __init__(self, gamma):
        self.gamma = gamma
    def __call__(self, n, L):
        sum_list = sum([pow(self.gamma, 2 * (i-1) ) for i in range(1,n)])
        numerator = sum_list ** -1
        den = sum([sum_list ** -1 for _ in range(n, L)])

        return numerator / den

def sarsa_gamma():
    env = gym.make('MountainCar-v0')

    def epsilon_greedy_policy(s, done, w, epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s, done, a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    alpha = 1e-4
    gamma = 1

    num_episodes = 20

    nA = env.action_space.n

    X = StateActionFeatureVectorWithTile(
        env.observation_space.low,
        env.observation_space.high,
        env.action_space.n,
        num_tilings=10,
        tile_width=np.array([.45, .035])
    )

    w = np.zeros((X.feature_vector_len()))

    gen = GenEstimator(gamma)
    phi_list = []

    for eps in range(num_episodes):
        #print('episode #{}'.format(eps))
        traj_list = []
        reward_list = []

        state, reward, done = env.reset(), 0, False

        action = epsilon_greedy_policy(state, done, w)

        traj_list.append((state, reward, action, done))

        T = 0

        while not done:
            next_state, reward, done, _ = env.step(action)
            action = int(epsilon_greedy_policy(next_state, done, w))
            traj_list.append((next_state, reward, action, done))
            # t+=1
            T += 1

        # state, reward, done = env.reset(), 0, False

        # action = epsilon_greedy_policy(state, done, w)

        phi_0 = X(state, done, action)
        phi_list.append(phi_0)

        reward_list.append(reward)

        for u in range(1, T):
            state, reward, action, done = traj_list[u]

            phi_u = X(state, done, action)
            phi_list.append(phi_u)
            reward_list.append(reward)

            delta = np.zeros((X.feature_vector_len()))

            for t in range(0, u - 1):
                # print(t)
                # print(u)
                # print(len(reward_list))

                # env.render()

                state, reward, action, done = traj_list[t]

                phi_t = X(state, done, action)

                a = phi_t - pow(gamma, u - t) * phi_u
                b = sum([pow(gamma, i - t) * reward_list[i]
                         for i in range(t, u - 1)])

                delta = delta + gen(u - t, T - t) * ((np.dot(w, a) - b)) * phi_t

                # print(delta)
            w = w - alpha * delta
            # print(w)

        phi_list = []
        reward_list = []
        # t = 0
        # T = 200
    return w