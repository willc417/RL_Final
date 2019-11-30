import numpy as np 
import gym 
from time import sleep

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        # TODO: implement here

        self.num_actions = num_actions
        self.num_tilings = num_tilings

        self.state_low = state_low 
        self.state_high = state_high

        self.tile_width = tile_width

        #print(self.tile_width)
        self.tile_size = np.ceil((self.state_high - self.state_low) / self.tile_width) + 1  
        print(self.tile_size)
        self.ntiles = self.tile_size[0] * self.tile_size[1]
        self.tile_starts = np.zeros((self.num_tilings, self.state_high.shape[0]))
        
        #print(self.weights.shape)
        for i in range(0, self.num_tilings):
            #print(self.state_low - i // (self.num_tilings * self.tile_width))
            self.tile_starts[i] = (self.state_low - i /  self.num_tilings * self.tile_width)


        print(self.tile_starts)
    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        # TODO: implement this method
        return int(self.num_tilings * self.ntiles * self.num_actions)

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        # TODO: implement this method


        if not done: 
            feats = np.zeros((self.feature_vector_len()))    
        else:
            feats = np.zeros((self.feature_vector_len()))
            return feats
    

        for i in range(0, self.num_tilings):
            tile = (s - self.tile_starts[i]) // self.tile_width
            feats[i*36 +(6*int(tile[0])) + (int(tile[1])) + int(a)] = 1 

        return feats


# equation 16 

class GenEstimator():
    def __init__(self, gamma):
        self.gamma = gamma
    def __call__(self, n, L):
        sum_list = sum([pow(self.gamma, 2 * (i-1) ) for i in range(1,n)])
        numerator = sum_list ** -1 
        den = sum([sum_list ** -1 for _ in range(n, L)])

        return numerator / den 


def test_sarsa_lamda(w):
    env = gym.make("MountainCar-v0")
    gamma = 1.

    X = StateActionFeatureVectorWithTile(
        env.observation_space.low,
        env.observation_space.high,
        env.action_space.n,
        num_tilings=10,
        tile_width=np.array([.45,.035])
    )

    def greedy_policy(s,done):
        Q = [np.dot(w, X(s,done,a)) for a in range(env.action_space.n)]
        return np.argmax(Q)

    def _eval(render=False):
        #print('hello')
        s, done = env.reset(), False
        if render: env.render()

        G = 0.
        while not done:
            a = greedy_policy(s,done)
            s,r,done,_ = env.step(a)
            if render: env.render()

            G += r
        return G

    Gs = [_eval() for _ in  range(100)]
    _eval(False)

    print(np.max(Gs))
    assert np.max(Gs) >= -110.0, 'fail to solve mountaincar'




def retrace_gamma():
    env = gym.make('MountainCar-v0')
    def epsilon_greedy_prob(epsilon, w, done,a):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        rand = np.random.random()
        #print(rand)
        prob = 0 
        if rand < epsilon:
            action = vals.argmax()
            prob = 1 
        else:
            action = env.action_space.sample()
            prob =  (1/3) * epsilon + (1/3)
            print('sss: {}'.format(prob))
        return int(action), prob

    def expectation_action_all_q(a, w, s, done, prob):
        expected_value = 0 
        curr_other_prob = (1 - prob) / 2
        for a in range(env.action_space.n):
            
            if a == action_taken:
                n_prob = prob
                #print(prob)
                val = np.dot(w, X(s,done,a))
                expected_value += n_prob * val
            else: 
                print(curr_other_prob)
                expected_value += curr_other_prob * val

    nA = env.action_space.n

    X = StateActionFeatureVectorWithTile(
        env.observation_space.low,
        env.observation_space.high,
        env.action_space.n,
        num_tilings=10,
        tile_width=np.array([.45,.035])
    )

    w = np.zeros((X.feature_vector_len()))

    gen = GenEstimator(gamma)
    phi_list = []


    for eps in range(num_episodes):
        print('episode #{}'.format(eps))
        traj_list = []
        reward_list = []

        state, reward, done = env.reset(), 0, False 

        action = epsilon_greedy_policy(state, done, w)

        traj_list.append((state, reward, action, done))

        T = 0
       
        while not done:

            next_state, reward, done, _ = env.step(action)
            action = int(epsilon_greedy_policy(next_state, done ,w))
            traj_list.append((next_state, reward, action, done))
            #t+=1 
            T += 1
        
        #state, reward, done = env.reset(), 0, False 

        #action = epsilon_greedy_policy(state, done, w)

        phi_0 = X(state, done, action)
        phi_list.append(phi_0)

        reward_list.append(reward)

        for u in range(1, T):
            state, reward, action, done = traj_list[u]

            phi_u  = X(state, done, action)
            phi_list.append(phi_u)
            reward_list.append(reward)

            delta = np.zeros((X.feature_vector_len()))

            for t in range(0, u-1):
                #print(t)
                #print(u)
                #print(len(reward_list))

                #env.render()

                state, reward, action, done = traj_list[t]

                phi_t = X(state, done, action)
                
                a = phi_t - pow(gamma, u-t) * phi_u
                b = sum([pow(gamma, i-t) *  reward_list[i]
                            for i in range(t, u-1)   ])


                delta = delta + gen(u -t, T - t) * ((a -b)) * phi_t
                
                #print(delta)
            w = w - alpha * delta
            #print(w)

        phi_list = [] 
        reward_list = []
        #t = 0 
        #T = 200 
    return w








def sarsa_gamma():
    env = gym.make('MountainCar-v0')
    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    gamma = .5
    alpha = 0.01 

    num_episodes = 10
    
    nA = env.action_space.n

    X = StateActionFeatureVectorWithTile(
        env.observation_space.low,
        env.observation_space.high,
        env.action_space.n,
        num_tilings=10,
        tile_width=np.array([.45,.035])
    )

    w = np.zeros((X.feature_vector_len()))

    gen = GenEstimator(gamma)
    phi_list = []

    for eps in range(num_episodes):
        print('episode #{}'.format(eps))
        traj_list = []
        reward_list = []

        state, reward, done = env.reset(), 0, False 

        action = epsilon_greedy_policy(state, done, w)

        traj_list.append((state, reward, action, done))

        T = 0
       
        while not done:

            next_state, reward, done, _ = env.step(action)
            action = int(epsilon_greedy_policy(next_state, done ,w))
            traj_list.append((next_state, reward, action, done))
            #t+=1 
            T += 1
        
        #state, reward, done = env.reset(), 0, False 

        #action = epsilon_greedy_policy(state, done, w)

        phi_0 = X(state, done, action)
        phi_list.append(phi_0)

        reward_list.append(reward)

        for u in range(1, T):
            state, reward, action, done = traj_list[u]

            phi_u  = X(state, done, action)
            phi_list.append(phi_u)
            reward_list.append(reward)

            delta = np.zeros((X.feature_vector_len()))

            for t in range(0, u-1):
                #print(t)
                #print(u)
                #print(len(reward_list))

                #env.render()

                state, reward, action, done = traj_list[t]

                phi_t = X(state, done, action)
                
                a = phi_t - pow(gamma, u-t) * phi_u
                b = sum([pow(gamma, i-t) *  reward_list[i]
                            for i in range(t, u-1)   ])


                delta = delta + gen(u -t, T - t) * ((a -b)) * phi_t
                
                #print(delta)
            w = w - alpha * delta
            #print(w)

        phi_list = [] 
        reward_list = []
        #t = 0 
        #T = 200 
    return w
      
w = sarsa_gamma()
test_sarsa_lamda(w)


