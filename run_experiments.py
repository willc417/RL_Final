from test_sarsa_gamma import test_sarsa_gamma
from test_retrace_gamma import test_retrace_gamma
from test_sarsa_lambda import test_sarsa_lambda
import matplotlib.pyplot as plt
import pandas as pd


def main():
    print("Running Sarsa Gamma")
    sarsa_gamma_df, sarsa_gamma_rpe = test_sarsa_gamma()
    sarsa_gamma_df.plot(kind='line', x='Gamma Values', y='Max Rewards', title="Sarsa Gamma Results")
    print(sarsa_gamma_rpe)
    plt.savefig('sarsa_gamma_plot.png')
    sarsa_gamma_rpe.plot(kind='line', use_index=True,  y='Rewards Per Episode', title="Sarsa Gamma RPE")


    print("Running Sarsa Lambda")
    sarsa_lambda_df, sarsa_lambda_rpe = test_sarsa_lambda()
    sarsa_lambda_df.plot(kind='line', x='Lambda Values', y='Max Rewards', title="Sarsa Lambda Results")
    plt.savefig('sarsa_lambda_plot.png')

    print(sarsa_lambda_rpe)



    #print("Running Retrace Gamma")
    #retrace_gamma_df = test_retrace_gamma()
    #retrace_gamma_df.plot(kind='line', x='Gamma Values', y='Max Rewards', title="Retrace Gamma Results")
    #plt.savefig('retrace_gamma_plot.png')

    #plt.plot('Gamma Values', 'Max Rewards', data=sarsa_gamma_df, marker='o', markerfacecolor='blue', linewidth=2, markersize=12, color='skyblue', label="Sarsa Gamma")
    #plt.plot('Lambda Values', 'Max Rewards', data=sarsa_lambda_df, marker='v', markerfacecolor='olive', markersize=12, color='olive', linewidth=2, label="Sarsa Lambda")
    #plt.plot('Gamma Values', 'Max Rewards', data=retrace_gamma_df, marker='x', markerfacecolor='red', markersize=12, color='red', linewidth=2, linestyle='dashed', label="Retrace Gamma")
    #plt.legend()
    #plt.savefig('combined_plot.png')
    #plt.show()


if __name__ == "__main__":
    main()