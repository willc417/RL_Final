import numpy as np
import gym
from sarsa_lambda import SarsaLambda, StateActionFeatureVectorWithTile

def test_sarsa_lamda():

    env = gym.make("MountainCar-v0")
    gamma = 1.
    print(env.observation_space.low)
    print(env.observation_space.high)

    X = StateActionFeatureVectorWithTile(
        env.observation_space.low,
        env.observation_space.high,
        env.action_space.n,
        num_tilings=10,
        tile_width=np.array([.45, .035]) #[.45, 50, .45, 50]) #Need to adjust the tile width for CartPole
    )

    w = SarsaLambda(env, gamma, 0.8, 0.01, X, 200)

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

    Gs = [_eval() for _ in  range(100)]
    _eval(False)
    print(np.max(Gs))
    #assert np.max(Gs) >= -110.0, 'fail to solve mountaincar'

if __name__ == "__main__":
    test_sarsa_lamda()
