import numpy as np
import gym
#import pandas as pd
from time import sleep
from StateActionFeatureVector import StateActionFeatureVectorWithTile

#np.random.seed(3258415304)
np.random.seed(1000)
#print(np.random.get_state()[1][0])

class GenEstimator():
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, n, L):
        sum_list = sum([pow(self.gamma, 2 * (i - 1)) for i in range(1, n)])
        numerator = sum_list ** -1
        den = sum([sum_list ** -1 for _ in range(n, L)])

        return numerator / den


def retrace_gamma(num_episodes, gamma):
    #gamma = 1
    reward_list = []
    def epsilon_greedy_prob(epsilon, w, s, done):
        nA = env.action_space.n
        Q = [np.dot(w, X(s, done, a)) for a in range(nA)]
        Q = np.array(Q)

        rand = np.random.random()
        # print(rand)
        prob = 0
        if rand > epsilon:

            action = Q.argmax()
            prob = 1
        else:
            action = env.action_space.sample()
            prob = (1 / nA) * epsilon + (1 / nA)
        return int(action), prob

    def target_policy(w, s, done):
        nA = env.action_space.n
        Q = [np.dot(w, X(s, done, a)) for a in range(nA)]
        Q = np.array(Q)

        action = np.argmax(Q)
        return int(action), 1

    def all_actions_state_feature(s, w , done):
        nA = env.action_space.n

        Q = [np.dot(w, X(s, done, a)) for a in range(nA)]
        Q = np.array(Q)

        action = np.argmax(Q)

        agg_feat = 0
        for a in range(nA):
            if a == action:
                agg_feat += 1 * X(s, done, a)
            else:
                agg_feat += 0 * X(s, done, a)

        return agg_feat

    def all_actions_qvals(s, w , done):
        nA = env.action_space.n

        Q = [np.dot(w, X(s, done, a)) for a in range(nA)]
        Q = np.array(Q)

        action = np.argmax(Q)

        agg_feat = 0
        for a in range(nA):
            if a == action:
                agg_feat += 1 * Q[a]
            else:
                agg_feat += 0 * Q[a]

        return agg_feat


    env = gym.make('MountainCar-v0')

    epsilon = .1
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
        prob_list = []
        tar_prob_list = []

        state, reward, done = env.reset(), 0, False

        action, prob = epsilon_greedy_prob(epsilon, w, state, done)
        tar_action, tar_prob = target_policy(w, state, done)

        if tar_action != action:
            tar_prob = 0

        

        traj_list.append((state, reward, action, done, prob, tar_prob))

        T = 0
        
        while not done:

            next_state, reward, done, _ = env.step(action)
            total_rewards += reward
            #reward_list.append(reward)
            action, prob = epsilon_greedy_prob(epsilon, w, next_state, done)
            tar_action, tar_prob = target_policy(w, state, done)

            if tar_action != action:
                tar_prob = 0
            traj_list.append((next_state, reward, action, done, prob, tar_prob))
            # t+=1
            T += 1

        if eps % 10 == 0 and eps != 0:
            rewards_per_episode.append(total_rewards / 10)
            total_rewards = 0

        phi_0 = X(state, done, action)
        phi_list.append(phi_0)

        reward_list.append(reward)
        prob_list.append(prob)

        for u in range(1, T):
            state, reward, action, done, prob, tar_prob = traj_list[u]

            phi_u = X(state, done, action)
            reward_list.append(reward)
            prob_list.append(prob)
            tar_prob_list.append(tar_prob)

            delta = np.zeros((X.feature_vector_len()))
            #trunc_is

            for t in range(0, u - 1):
                # print(t)
                # print(u)
                # print(len(reward_list))

                # env.render()
                # print(traj_list[t])
                state, reward, action, done, prob, tar_prob = traj_list[t]

                phi_t = X(state, done, action)
                #phi_all_t = all_actions_state_feature(state, w, done)
                qvals = all_actions_qvals(state, w, done)

                im_s = np.prod([1 / prob_list[i]
                                for i in range(t, u - 1)])

                trunc_is = min(1, im_s)

                #a = (trunc_is*(phi_all_t - phi_t)) - (pow(gamma, u - t) * phi_u)
                a = phi_t - (pow(gamma, u - t) * phi_u)
                b = sum([pow(gamma, i - t) * reward_list[i]
                         for i in range(t, u - 1)])

                # print([ (tar_prob_list[i], prob_list[i])
                #                    for i in range(t, u-1) ])

                if eps > 50:
                    delta = delta + gen(u - t, T - t) * \
                            (( trunc_is * (np.dot(w, a) - b+qvals) ) * phi_t)
                else:
                    delta = delta + gen(u - t, T - t) * \
                            (((np.dot(w, a) - b+qvals) ) * phi_t)

                # print(delta)
            w = w - alpha * delta
            # print(w)

        phi_list = []
        reward_list = []

    return w, rewards_per_episode

