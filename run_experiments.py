#!/usr/bin/env python3

from test_sarsa_gamma import test_sarsa_gamma
from test_retrace_gamma import test_retrace_gamma
from test_sarsa_lambda import test_sarsa_lambda
from test_retrace_lambda import test_retrace_lambda
import matplotlib.pyplot as plt
import pandas as pd


def main():
    num_episodes = 0
    while num_episodes != 50 and num_episodes != 500 and num_episodes != 1000 and num_episodes != 5000:
        num_episodes = int(input("Welcome. How many episodes would you like to train on? (100, 500, 1000, 5000) "))

    print("Running Sarsa Gamma")
    sarsa_gamma_df, sarsa_gamma_rpe = test_sarsa_gamma(num_episodes = num_episodes)
    #sarsa_gamma_df.plot(kind='line', x='Gamma Values', y='Max Rewards', title="Sarsa Gamma Results")

    print("Running Sarsa Lambda")
    sarsa_lambda_df, sarsa_lambda_rpe = test_sarsa_lambda(num_episodes = num_episodes)
    #sarsa_lambda_df.plot(kind='line', x='Lambda Values', y='Max Rewards', title="Sarsa Lambda Results")
    #plt.savefig('sarsa_lambda_plot.png')

    sarsa_lambda_rpe = sarsa_lambda_rpe.join(sarsa_gamma_rpe)
    srsa = sarsa_lambda_rpe.plot(style='.-', title="Sarsa Lambda and Sarsa Gamma")
    srsa.set_xlabel('Episode / 10')
    srsa.set_ylabel('Reward Per 10 episodes')
    plt.savefig("sarsa_lambda_and_gamma.png")
    #plt.show()

    print("Running Retrace Gamma")
    retrace_gamma_df, retrace_gamma_rpe = test_retrace_gamma(num_episodes = num_episodes)
    #retrace_gamma_df.plot(kind='line', x='Gamma Values', y='Max Rewards', title="Retrace Gamma Results")
    #plt.savefig('retrace_gamma_plot.png')

    print("Running Retrace Lambda")
    retrace_lambda_df, retrace_lambda_rpe = test_retrace_lambda(num_episodes=num_episodes)
    #retrace_lambda_df.plot(kind='line', x='Lambda Values', y='Max Rewards', title="Retrace Lambda Results")
    #plt.savefig('retrace_lambda_plot.png')

    retrace_lambda_rpe = retrace_lambda_rpe.join(retrace_gamma_rpe)
    retrace = retrace_lambda_rpe.plot(style='.-', title="Retrace Lambda and Gamma Rewards Per Episode")
    retrace.set_xlabel('Episode / 10')
    retrace.set_ylabel('Reward Per 10 episodes')
    plt.savefig("retrace_lambda_and_gamma.png")
    plt.show()

if __name__ == "__main__":
    main()