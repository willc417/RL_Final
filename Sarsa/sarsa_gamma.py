import numpy as np
import gym
from Feature_Representation.StateActionFeatureVector import StateActionFeatureVectorWithTile

np.random.seed(1000)

class GenEstimator():
    def __init__(self, gamma):
        self.gamma = gamma
    def __call__(self, n, L):
        sum_list = sum([pow(self.gamma, 2 * (i-1) ) for i in range(1,n)])
        numerator = sum_list ** -1
        den = sum([sum_list ** -1 for _ in range(n, L)])

        return numerator / den

def sarsa_gamma(num_episodes, gamma):
    env = gym.make('MountainCar-v0')

    def epsilon_greedy_policy(s, done, w, epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s, done, a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    alpha = 1e-4

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
    rewards_per_episode = []
    total_rewards = 0
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
            total_rewards += reward
            action = int(epsilon_greedy_policy(next_state, done, w))
            traj_list.append((next_state, reward, action, done))
            T += 1

        if (eps + 1) % 10 == 0 and eps != 0:
            rewards_per_episode.append(total_rewards / 10)
            total_rewards = 0

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
                
                state, reward, action, done = traj_list[t]

                phi_t = X(state, done, action)

                if t != u:
                    s, _, a, _ = traj_list[t+1]
                    phi_t_next = X(s, done, a)
                    #Q_next = np.dot(w, X(s, done, a))
                else:
                    phi_t_next = np.zeros(phi_t.shape)
                    Q_next = 0

                a = phi_t - pow(gamma, u - t) * phi_u
                a_prime = phi_t_next - pow(gamma, u - t) * phi_u
                b = sum([pow(gamma, i - t) * reward_list[i]
                         for i in range(t, u - 1)])

                delta = delta + gen(u - t, T - t) * ((np.dot(w, a) - (b+np.dot(w, a_prime)))) * phi_t

            w = w - alpha * delta
        phi_list = []

    return w, rewards_per_episode