import matplotlib.pyplot as plt
import pandas as pd

def main():

    #print("Running Sarsa Lambda")
    sarsa_lambda_rpe = pd.read_csv("sarsa_lambda_rewards_per_episode_used_for_plot.csv")
    sarsa_gamma_rpe = pd.read_csv("sarsa_gamma_rpe_used.csv")
    print(sarsa_lambda_rpe)
    sarsa_lambda_rpe = sarsa_lambda_rpe.drop(labels='1', axis=1)
    sarsa_lambda_rpe = sarsa_lambda_rpe.join(sarsa_gamma_rpe)
    srsa = sarsa_lambda_rpe.plot(title="Sarsa Lambda and Sarsa Gamma Training Reward Values")
    srsa.set_xlabel('Episode / 10')
    srsa.set_ylabel('Reward Per 10 episodes')

    srsa.legend(loc=4)

    plt.savefig("sarsa_lambda.png")
    plt.show()

    sarsa_l_max_min = pd.read_csv("sarsa_lambda_returns_used.csv")
    sarsa_g_max_min = pd.read_csv("sarsa_gamma_returns_used.csv")

    sarsa_l_max_min = sarsa_l_max_min.rename(columns={"Lambda Values": "Lambda Values (1.0 Represents Sarsa Gamma)"})
    sarsa_g_max_min = sarsa_g_max_min.rename(columns={"Gamma Values": "Lambda Values (1.0 Represents Sarsa Gamma)"})
    sarsa_l_max_min = sarsa_l_max_min.append(sarsa_g_max_min)

    sarsa_l_max_min.plot.bar(x=0, title="Sarsa Lambda and Sarsa Gamma Evaluation Rewards")
    plt.show()

    print(sarsa_l_max_min)


    #
    # retrace_lambda_rpe = pd.read_csv("retrace_lambda_rewards_per_episode.csv")
    #
    # retrace_gamma_rpe = pd.read_csv("retrace_gamma_rpe.csv")
    #
    # retrace_lambda_rpe = retrace_lambda_rpe.join(retrace_gamma_rpe)
    # retrace = retrace_lambda_rpe.plot(style='.-', title="Retrace Lambda and Retrace Gamma")
    # retrace.set_xlabel('Episode / 10')
    # retrace.set_ylabel('Reward Per 10 episodes')
    # plt.savefig("retrace_lambda.png")
    # plt.show()
    #
    # sarsa_gamma_rpe = pd.read_csv("sarsa_gamma_rpe.csv")
    # sarsa_gamma_rpe = sarsa_gamma_rpe.rename(columns={'Gamma': 'Sarsa Gamma'})
    # retrace_gamma_rpe = pd.read_csv("retrace_gamma_rpe.csv")
    # retrace_gamma_rpe = retrace_gamma_rpe.rename(columns={'Gamma': 'Retrace Gamma'})
    # print(retrace_gamma_rpe)
    # gamma_combined = retrace_gamma_rpe.join(sarsa_gamma_rpe)
    # gamma_plt = gamma_combined.plot(style='.-', title="Retrace Gamma and Sarsa Gamma")
    # gamma_plt.set_xlabel('Episode / 10')
    # gamma_plt.set_ylabel('Reward Per 10 episodes')
    # plt.savefig("gamma_combined.png")
    # plt.show()


if __name__ == "__main__":
    main()