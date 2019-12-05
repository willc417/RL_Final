import numpy as np
from StateActionFeatureVector import StateActionFeatureVectorWithTile


def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len()))
    for episode in range(num_episode):
        done = False
        observation = env.reset()
        action = epsilon_greedy_policy(observation, done, w)
        action_x = X(observation, done, action)
        Q_old = 0
        z = 0
        for t in range(0, episode):
            observation, reward, done, info = env.step(action)
            next_action = epsilon_greedy_policy(observation, done, w)
            next_action_x = X(observation, done, next_action)
            Q = np.dot(w, action_x)
            Q_prime = np.dot(w, next_action_x)
            td = reward + gamma * Q_prime - Q
            z = gamma * lam * z + (1 - (alpha * gamma * lam * z) * np.transpose(action_x)) * action_x
            w = w + alpha * (td + Q - Q_old) * z - alpha * (Q - Q_old) * action_x
            Q_old = Q_prime
            action_x = next_action_x
            action = next_action
    return w
