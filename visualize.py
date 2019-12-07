import matplotlib.pyplot as plt
import pandas as pd

def main():

    #print("Running Sarsa Lambda")
    sarsa_lambda_rpe = pd.read_csv("sarsa_lambda_rewards_per_episode.csv")
    print(sarsa_lambda_rpe)

    sarsa_gamma_rpe = pd.read_csv("sarsa_gamma_rpe.csv")

    sarsa_lambda_rpe = sarsa_lambda_rpe.join(sarsa_gamma_rpe)
    sarsa_lambda_rpe.plot()
    plt.savefig("sarsa_lambda.png")
    plt.show()


if __name__ == "__main__":
    main()