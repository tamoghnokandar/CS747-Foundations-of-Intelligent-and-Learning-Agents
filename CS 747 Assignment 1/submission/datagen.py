import numpy as np
import os
def epsilon_greedy(instance, arms, randomSeed, horizon, f, epsilon, scale = 2, threshold = 0, highs = 0):
    """
    :param instance:
    :param arms:
    :param randomSeed:
    :param horizon:
    :param f:
    :param epsilon:
    :param scale:
    :param threshold:
    :param highs:
    :return:
    """
    np.random.seed(randomSeed)
    n = len(arms) # number of arms
    REW = r = 0
    values = np.array([0.0 for i in range(n)])
    count = np.array([0 for i in range(n)])

    for t in range(horizon):
        if np.random.random() < epsilon:
            arm = np.random.randint(0, n)
        else:
            arm = np.random.choice(np.where(values==values.max())[0])

        r = np.random.binomial(1, arms[arm])
        count[arm] += 1
        values[arm] += (r - values[arm]) / count[arm]
        REW += r

    reg = horizon * max(arms) - REW

    f.write(instance + ", " + "epsilon-greedy-t1, " + str(randomSeed) + ", "+ str(epsilon)+ ", " \
            + str(scale)+ ", " + str(threshold)+ ", " + str(horizon)+ ", " + str(reg)+ ", "+ str(highs) + "\n")
    pass

def ucb_t1(instance, arms, randomSeed, horizon, f, epsilon = 0.02, scale = 2, threshold = 0, highs = 0):
    """
    :param instance:
    :param arms:
    :param randomSeed:
    :param horizon:
    :param f:
    :param epsilon:
    :param scale:
    :param threshold:
    :param highs:
    :return:
    """
    np.random.seed(randomSeed)
    n = len(arms) # number of arms
    REW = r = 0
    values = np.array([0.0 for i in range(n)])
    count = np.array([0 for i in range(n)])

    for t in range(n):
        r = np.random.binomial(1, arms[t])
        count[t] += 1
        values[t] += r
        REW+=r

    ucbs = np.empty(n)
    for t in range(n, horizon):

        for i in range(n):
            ucbs[i] = values[i] + np.sqrt(float(scale) * np.log(t)/count[i])

        arm = np.random.choice(np.where(ucbs==ucbs.max())[0])
        r = np.random.binomial(1, arms[arm])
        count[arm] += 1
        values[arm] += (r - values[arm]) / count[arm]
        REW+=r

    reg = horizon * max(arms) - REW

    f.write(instance + ", " + "ucb-t1, " + str(randomSeed) + ", " + str(epsilon) + ", " \
            + str(scale) + ", " + str(threshold) + ", " + str(horizon) + ", " + str(reg) + ", " + str(highs) + "\n")
    pass

def ucb_t2(instance, arms, randomSeed, horizon, f, epsilon = 0.02, scale = 2, threshold = 0, highs = 0):
    """
    :param instance:
    :param arms:
    :param randomSeed:
    :param horizon:
    :param f:
    :param epsilon:
    :param scale:
    :param threshold:
    :param highs:
    :return:
    """
    np.random.seed(randomSeed)
    n = len(arms) # number of arms
    REW = r = 0
    values = np.array([0.0 for i in range(n)])
    count = np.array([0 for i in range(n)])

    for t in range(n):
        r = np.random.binomial(1, arms[t])
        count[t] += 1
        values[t] += r
        REW+=r

    ucbs = np.empty(n)
    for t in range(n, horizon):

        for i in range(n):
            ucbs[i] = values[i] + np.sqrt(float(scale) * np.log(t)/count[i])

        arm = np.random.choice(np.where(ucbs == ucbs.max())[0])
        r = np.random.binomial(1, arms[arm])
        count[arm] += 1
        values[arm] += (r - values[arm]) / count[arm]
        REW+=r

    reg = horizon * max(arms) - REW

    f.write(instance + ", " + "ucb-t2, " + str(randomSeed) + ", " + str(epsilon) + ", " \
            + str(scale) + ", " + str(threshold) + ", " + str(horizon) + ", " + str(reg) + ", " + str(highs) + "\n")
    pass


def KL(p, q):
    if p==q:
        return 0
    if q==0 or q==1:
        return float("inf")
    if p==1:
        return np.log(1/q)
    if p==0:
        return -np.log(1-q)
    return p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))

def get_kl_ucb(value, count, t, c=3):
    delta = 0.005
    target = (np.log(t) + c*np.log(np.log(t)))/count
    low, high = value, 1

    while high-low>=delta:
        mid = (low + high) / 2
        res = KL(value, mid)
        if target>res and target-res<=delta:
            return mid
        elif res > target:
            high = mid
        else:
            low = mid
    return low

def kl_ucb(instance, arms, randomSeed, horizon, f, epsilon = 0.02, scale = 2, threshold = 0, highs = 0):
    """
    :param instance:
    :param arms:
    :param randomSeed:
    :param horizon:
    :param f:
    :param epsilon:
    :param scale:
    :param threshold:
    :param highs:
    :return:
    """
    np.random.seed(randomSeed)
    n = len(arms) # number of arms
    REW = r = 0
    values = np.array([0.0 for i in range(n)])
    count = np.array([0 for i in range(n)])

    for t in range(n):
        r = np.random.binomial(1, arms[t])
        count[t] += 1
        values[t] += r
        REW+=r

    kl_ucbs = np.empty(n)
    for t in range(n, horizon):
        for i in range(n):
            kl_ucbs[i] = get_kl_ucb(values[i], count[i], t)
        arm = np.argmax(kl_ucbs)
        r = np.random.binomial(1, arms[arm])
        count[arm] += 1
        values[arm] += (r - values[arm]) / count[arm]
        REW+=r

    reg = horizon * max(arms) - REW

    f.write(instance + ", " + "kl-ucb-t1, " + str(randomSeed) + ", " + str(epsilon) + ", " \
            + str(scale) + ", " + str(threshold) + ", " + str(horizon) + ", " + str(reg) + ", " + str(highs) + "\n")
    pass

def thompson_sampling(instance, arms, randomSeed, horizon, file, epsilon = 0.02, scale = 2, threshold = 0, highs = 0):
    """
    :param instance:
    :param arms:
    :param randomSeed:
    :param horizon:
    :param f:
    :param epsilon:
    :param scale:
    :param threshold:
    :param highs:
    :return:
    """
    np.random.seed(randomSeed)
    n = len(arms) # number of arms
    REW = r = 0
    values = np.array([0.0 for i in range(n)])
    count = np.array([0 for i in range(n)])

    for t in range(horizon):

        arm = 0
        max_sampled = -1
        for i in range(n):
            s = values[i]*count[i]
            f = count[i]-s
            sampled = np.random.beta(s+1, f+1)
            if(sampled>max_sampled):
                arm = i
                max_sampled = sampled

        r = np.random.binomial(1, arms[arm])
        count[arm] += 1
        values[arm] += (r - values[arm]) / count[arm]
        REW+=r

    reg = horizon * max(arms) - REW

    file.write(instance + ", " + "thompson-sampling-t1, " + str(randomSeed) + ", " + str(epsilon) + ", " \
            + str(scale) + ", " + str(threshold) + ", " + str(horizon) + ", " + str(reg) + ", " + str(highs) + "\n")
    pass


path = "../instances/instances-task1/"
instances = []
for i in range(1,len(os.listdir(path))+1):
    instances.append(path + "i-" + str(i) + ".txt")
#print(instances)

#instances = os.listdir(path)
seeds = range(50)
algorithms = [epsilon_greedy, ucb_t1, kl_ucb, thompson_sampling]
epsilon = 0.02
scale_t1 = 2
threshold = 0
highs = 0
horizons = [100, 400, 1600, 6400, 25600, 102400]
for instance in instances:
    with open(instance) as f:
        arms = [line.strip() for line in f]
        arms = list(map(float, arms))

    with open("outputData.txt", "a") as f:
        for algo in algorithms:
            for seed in seeds:
                for horizon in horizons:
                    algo(instance, arms, seed, horizon, f, epsilon, scale_t1, threshold, highs)




path_t2 = "../instances/instances-task2/"
instances_t2 = []
for i in range(1,len(os.listdir(path_t2))+1):
    instances_t2.append(path_t2 + "i-" + str(i) + ".txt")
#print(instances)

#instances = os.listdir(path)

algorithms_t2 = [ucb_t2]
scales = [float(i/100) for i in range(2, 32, 2)]
horizon_t2 = 10000
for instance in instances_t2:
    with open(instance) as f:
        arms = [line.strip() for line in f]
        arms = list(map(float, arms))

    with open("outputData.txt", "a") as f:
        for algo in algorithms_t2:
            for seed in seeds:
                for s in scales:
                    algo(instance, arms, seed, horizon_t2, f, epsilon, s, threshold, highs)
