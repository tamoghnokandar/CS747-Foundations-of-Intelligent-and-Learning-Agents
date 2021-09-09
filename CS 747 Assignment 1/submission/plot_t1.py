import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("outputData.txt", header=None, delimiter=", ")
df.columns = ['instance', 'algorithm', 'seed', 'epsilon', 'scale', 'threshold', 'horizon', 'regret', 'highs']

data = df.groupby(['instance', 'algorithm', 'horizon']).mean().reset_index()


algos = ['epsilon-greedy-t1', 'ucb-t1', 'kl-ucb-t1', 'thompson-sampling-t1']
markers = ['o', 'x', 'd', '<']
path = "../instances/instances-task1/"
instances = []
for i in range(1, len(os.listdir(path))+1):
    instances.append(path + "i-" + str(i) + ".txt")

plt.rcParams['figure.figsize'] = 12, 8
for instance in instances:
    plt.xscale("log")
    for i in range(len(algos)):
        temp=data[(data.instance==instance) & (data.algorithm==algos[i])]
        x = list(temp.horizon)
        y = list(temp.regret)
        plt.plot(x,y,marker= markers[i], label=algos[i])
    plt.xlabel("Horizon", fontsize=12)
    plt.ylabel("Regret", fontsize=12)
    plt.title("Bandit Instance " + str(instance[-5]) + ": Regret vs Horizon" +"\n", fontsize=20)
    plt.legend()
    plt.savefig("../plots/T1_instance_" + str(instance[-5])+".png")
    plt.show()