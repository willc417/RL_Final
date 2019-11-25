# SARSA-lambda with Gaussian radial basis functions for action-value approximation
# Implemented for the OpenAI gym mountain-car environment
# Written by Evan Gravelle
# 7/28/2016

import gym
import numpy as np
import matplotlib.pyplot as plt

# Initializations
env = gym.make('CartPole-v0')
#env.monitor.start('./tmp/mountain-car-1', force=True)
num_actions = env.action_space.n
dim = env.observation_space.high.size
print(env.observation_space.high)

# Parameters
# one set which converges in around 1200 episodes
# 4 rows, 4 cols, eps = 0.1, Lambda = 0.5, alpha = 0.008, gamma = 0.99
num_rbf = 4 * np.ones(4).astype(int)
print(num_rbf)
width = 1. / (num_rbf - 1.)
print(width)
rbf_sigma = width[0] / 2.
epsilon = 0.2
epsilon_final = 0.1
Lambda = 0.9
alpha = 0.01
gamma = 0.99
num_episodes = 2000
num_timesteps = 200

xbar = np.zeros((4, dim))
xbar[0, :] = env.observation_space.low
xbar[1, :] = env.observation_space.high

print(xbar)
num_ind = np.prod(num_rbf)
activations = np.zeros(num_ind)
new_activations = np.zeros(num_ind)
theta = np.zeros((num_ind, num_actions))
rbf_den = 2 * rbf_sigma ** 2
epsilon_coefficient = (epsilon - epsilon_final) ** (1. / num_episodes)
ep_length = np.zeros(num_episodes)
np.set_printoptions(precision=2)


# Construct ndarray of rbf centers
c = np.zeros((num_ind, dim))
print(c.shape)
print(num_rbf)
for i in range(num_rbf[0]):
    print(i)
    for j in range(num_rbf[1]):
        for k in range(num_rbf[2]):
            for l in range(num_rbf[3]):
                c[i*num_rbf[1] + j, :] = (l*width[3],k * width[2],i * width[1], j * width[0])


# Returns the state scaled between 0 and 1
def normalize_state(s):
    y = np.zeros(len(s))
    for i in range(len(s)):
        y[i] = (s[i] - xbar[0, i]) / (xbar[1, i] - xbar[0, i])
    return y


# Returns an ndarray of radial basis function activations
def phi(state):
    phi = np.zeros(num_ind)
    for k in range(num_ind):
        phi[k] = np.exp(-np.linalg.norm(state - c[k, :]) ** 2 / rbf_den)
    return phi


# Returns an action following an epsilon-greedy policy


def epsilon_greedy_prob(epsilon, vals):
    rand = np.random.random()
    #print(rand)
    prob = 0 
    if rand < 1. - epsilon:
        action = vals.argmax()
        prob = 1 
    else:
        action = env.action_space.sample()
        prob =  (1/2) * epsilon + (1/2)
        #print('sss: {}'.format(prob))
    return int(action), prob


def expectation_action_all_q(action_taken, activations, theta, prob):
    expected_value = 0 
    curr_other_prob = (1 - prob) / 2
    for a in range(env.action_space.n):
        
        if a == action_taken:
            n_prob = prob
            print(prob)
            expected_value += n_prob * action_value(activations, action_taken, theta)

        else: 
            print(curr_other_prob)
            expected_value += curr_other_prob * action_value(activations, action_taken, theta)



    return expected_value 

def behavior_policy():
    return int(env.action_space.sample())


# Returns the value of each action at some state
def action_values(activations, theta):
    val = np.dot(theta.T, activations)
    return val



# Returns the value of an action at some state
def action_value(activations, action, theta):
    val = np.dot(theta[:, action], activations)
    return val


# SARSA loop
for ep in range(num_episodes):

    e = np.zeros((num_ind, num_actions))
    state = normalize_state(env.reset())

    #print(state)
    activations = phi(state)

    #print(activations)
    #print(activations.shape)
    #print(theta.shape)
    #print(theta)
    # print "activations = ", np.reshape(activations.ravel(order='F'), (num_rows, num_cols))
    vals = action_values(activations, theta)

    #print(vals)
    action, prob = epsilon_greedy_prob(epsilon, vals)

    Q = action_value(activations, action, theta)

    # Each episode
    for t in range(num_timesteps):

        # render, get activations for state, get next_state
        env.render()
        next_state, reward, done, info = env.step(action)
        new_state = normalize_state(next_state)
        new_activations = phi(new_state)

        
        # calculate Q and get next action
        new_vals = action_values(new_activations, theta)
        new_action, prob = epsilon_greedy_prob(epsilon, new_vals)
        Q_next  = action_value(new_activations, action, theta)

        # Calculate the expectation according to equation 3 in Retrace paper
        tar_exp = expectation_action_all_q(new_action, new_activations, theta, prob)
        
        
        #Q_new = action_value(new_activations, new_action, theta)
        # calculate td
        if done:
            td = reward - Q
        else:
            td = reward + (gamma * tar_exp) - Q
        
        # eligibility trace
        e[:, action] = activations 

        for k in range(num_ind):
            for a in range(num_actions):
    
                trunc_is = Lambda * min(1, 1 /prob)
                theta[k, a] += alpha * td * trunc_is * e[k, a]
        e *= gamma * Lambda

        if t % 1 == 0:
            print("t = ", t)
            print("new_state = ", new_state)
            #print("new_activations = ", np.reshape(new_activations.ravel(order='F'), (num_rows, num_cols)))
            #print("new_vals", new_vals)
            #print("Q = ", Q)
            #print("Q_new = ", Q_new)
            print("action = ", action)
            print("target = ", td)
            #print("e =", e)
            #print("theta = \n", np.reshape(theta.ravel(order='F'), (num_actions, num_rows, num_cols)))
            print("---------------------------------------------------------------------------")

        state = new_state.copy()
        activations = new_activations.copy()
        action = new_action
        print(action)
        Q = action_value(new_activations, new_action, theta)
        if done:
            break

    ep_length[ep] = t
    # print "t = ", t
    epsilon *= epsilon_coefficient
