import numpy as np
from StateActionFeatureVector import StateActionFeatureVectorWithTile

np.random.seed(1000)


def RetraceLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]
        prob = 0 
        if np.random.rand() < epsilon:
            prob = 1/nA * epsilon + 1 / nA
            return np.random.randint(nA), prob
        else:
            prob = 1
            return np.argmax(Q), prob


    def expectation_of_Q(s, w, done):
        nA = env.action_space.n

        Q = [np.dot(w, X(s, done, a)) for a in range(nA)]
        Q = np.array(Q)

        action = np.argmax(Q)

        agg_feat = 0
        prob = 0 
        for a in range(nA):
            if a == action:
                agg_feat += 1 * Q[a]
                prob = 1 
            else:
                agg_feat += 0 * Q[a]
                

        return agg_feat, prob
    
    rewards_per_episode = []
    
    total_reward = 0
    w = np.zeros((X.feature_vector_len()))
    for episode in range(num_episode):
        done = False
        observation = env.reset()
        action, prob = epsilon_greedy_policy(observation, done, w)
        action_x = X(observation, done, action)
        Q_old = 0
        z = 0
        #t = 0

        while not done: #for t in range(0, episode):
            observation, reward, done, info = env.step(action)
            total_reward += reward
            #reward_list.append(total_reward)
            next_action, prob = epsilon_greedy_policy(observation, done, w)
            next_action_x = X(observation, done, next_action)
            Q = np.dot(w, action_x)
            Q_prime = np.dot(w, next_action_x)
            Q_prime, tar_prob = expectation_of_Q(observation, w, done)

            trunc_is = lam * min(1, tar_prob/prob)

            td = trunc_is * (reward + gamma * Q_prime - Q)
            z = gamma * lam * z + (1 - (alpha * gamma * lam * z) * np.transpose(action_x)) * action_x
            w = w + alpha * (td + Q - Q_old) * z - alpha * (Q - Q_old) * action_x
            Q_old = Q_prime
            action_x = next_action_x
            action = next_action
        if episode % 10 == 0 and episode != 0:
            rewards_per_episode.append(total_reward / 10)
            total_reward = 0
    return w, rewards_per_episode
