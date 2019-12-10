#!/usr/bin/env python3

from Test.test_sarsa_gamma import test_sarsa_gamma
from Test.test_retrace_gamma import test_retrace_gamma
from Test.test_sarsa_lambda import test_sarsa_lambda
from Test.test_retrace_lambda import test_retrace_lambda
import matplotlib.pyplot as plt


def main():
    num_episodes = 0
    while num_episodes != 100 and num_episodes != 500 and num_episodes != 1000 and num_episodes != 5000:
        num_episodes = int(input("Welcome. How many episodes would you like to train on? (100, 500, 1000, 5000) "))

    print("Running Sarsa Gamma")
    sarsa_gamma_df, sarsa_gamma_rpe = test_sarsa_gamma(num_episodes = num_episodes)

    print("Running Sarsa Lambda")
    sarsa_lambda_df, sarsa_lambda_rpe = test_sarsa_lambda(num_episodes = num_episodes)
    #
    sarsa_lambda_rpe = sarsa_lambda_rpe.join(sarsa_gamma_rpe)
    srsa = sarsa_lambda_rpe.plot(title="Sarsa Lambda and Sarsa Gamma")
    srsa.set_xlabel('Episode / 10', fontsize=15)
    srsa.set_ylabel('Reward Per 10 episodes', fontsize=15)
    #plt.show()
    #
    sarsa_lambda_df = sarsa_lambda_df.rename(columns={"Lambda Values": "Lambda Values (1.0 Represents Sarsa Gamma)"})
    sarsa_gamma_df = sarsa_gamma_df.rename(columns={"Gamma Values": "Lambda Values (1.0 Represents Sarsa Gamma)"})
    sarsa_lambda_df = sarsa_lambda_df.append(sarsa_gamma_df)
    sarsa_lambda_df.plot.bar(x=0, title="Sarsa Lambda and Sarsa Gamma Evaluation Rewards")
    #plt.show()
    #
    #
    #
    print("Running Retrace Gamma - This will take a long time")
    retrace_gamma_df, retrace_gamma_rpe = test_retrace_gamma(num_episodes=num_episodes)
    #
    bar_retrace_gamma_df = retrace_gamma_df
    #
    #
    print("Running Retrace Lambda")
    retrace_lambda_df, retrace_lambda_rpe = test_retrace_lambda(num_episodes=num_episodes)
    #
    bar_retrace_lambda_df = retrace_lambda_df
    #
    retrace_lambda_rpe = retrace_lambda_rpe.join(retrace_gamma_rpe)
    retrace = retrace_lambda_rpe.plot(title="Retrace Lambda and Gamma Rewards Per Episode")
    retrace.set_xlabel('Episode / 10', fontsize=15)
    retrace.set_ylabel('Reward Per 10 episodes', fontsize=15)
    #plt.show()

    print(bar_retrace_lambda_df)
    print(bar_retrace_gamma_df)
    bar_retrace_lambda_df = bar_retrace_lambda_df.rename(
        columns={"Lambda Values": "Lambda Values (1.0 Represents Retrace Gamma)"})
    bar_retrace_gamma_df = bar_retrace_gamma_df.rename(columns={"Gamma Values": "Lambda Values (1.0 Represents Retrace Gamma)"})

    bar_retrace_lambda_df = bar_retrace_lambda_df.append(bar_retrace_gamma_df)
    print(bar_retrace_lambda_df)
    bar_retrace_lambda_df.plot.bar(x=0, title="Retrace Lambda and Retrace Gamma Evaluation Rewards")
    plt.show()


if __name__ == "__main__":
    main()