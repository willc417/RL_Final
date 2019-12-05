from StateActionFeatureVector import StateActionFeatureVectorWithTile
from retrace_gamma import retrace_gamma
import numpy as np
import gym


def test_retrace_gamma():
    env = gym.make("MountainCar-v0")

    w = retrace_gamma()
    gamma_values = []
    return_values = []

    gamma = 1.

    X = StateActionFeatureVectorWithTile(
        env.observation_space.low,
        env.observation_space.high,
        env.action_space.n,
        num_tilings=20,
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
    return_values.append(np.max(Gs))
    # assert np.max(Gs) >= -110.0, 'fail to solve mountaincar'
    # sarsa_gamma_data = pd.DataFrame(data={"Gamma Value": gamma_values, "Max Record": return_values})
    # sarsa_gamma_data.to_csv("sarsa_gamma_returns.csv")


if __name__ == "__main__":
    test_retrace_gamma()