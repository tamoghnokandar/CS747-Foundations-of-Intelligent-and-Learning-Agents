import pandas as pd
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    df = pd.read_csv("outputData_T2.txt", header=None, delimiter=", ")
    df.columns = ['instance', 'algorithm', 'seed', 'epsilon', 'scale', 'threshold', 'horizon', 'regret', 'highs']

    data = df.groupby(['instance', 'scale']).mean().reset_index()


    algos = ['ucb']
    markers = ['o', 'x', '*', 'D', '>']
    path = "../instances/instances-task2/"
    instances = []
    for i in range(1, len(os.listdir(path))+1):
        instances.append(path + "i-" + str(i) + ".txt")

    plt.rcParams['figure.figsize'] = 12, 8

    #plt.xscale("log")
    for i in range(len(instances)):

        temp = data[(data.instance == instances[i])]
        x = list(temp.scale)
        y = list(temp.regret)
        plt.plot(x, y, marker=markers[i], label=str('instance'+ instances[i][-5] ))

    plt.xlabel("Scale", fontsize=12)
    plt.ylabel("Regret", fontsize=12)
    plt.title("Regret vs Scale" + "\n", fontsize=20)
    plt.legend()
    plt.savefig("../plots/T2" + ".png")
    plt.show()