import sys
import numpy as np

def epsilon_greedy(instance, arms, randomSeed, horizon, epsilon, scale = 2, threshold = 0, highs = 0):
    """
    :param arms: The actual means of the bandit arms
    :param randomSeed: The random seed to generate pseudo random results
    :param horizon: The number of pulls the bandit should make
    :param epsilon: The exploration parameter
    :return: String of form "algorithm, random seed, epsilon, horizon, REG"
    """
    np.random.seed(randomSeed)
    num_arms = len(arms) # number of arms
    REW = 0
    rewards = np.zeros(num_arms)
    count = np.zeros(num_arms)

    for time in range(horizon):
        if np.random.random() < epsilon:
            arm = np.random.randint(0, num_arms)
        else:
            arm = np.random.choice(np.where(rewards == rewards.max())[0])

        current_reward = np.random.binomial(1, arms[arm])
        count[arm] += 1
        rewards[arm] += (current_reward - rewards[arm]) / count[arm]
        REW += current_reward

    reg = horizon * max(arms) - REW
    result = str(instance + ", " + "epsilon-greedy-t1, " + str(randomSeed) + ", "+ str(epsilon)+ ", " \
            + str(scale)+ ", " + str(threshold)+ ", " + str(horizon)+ ", " + str(reg)+ ", "+ str(highs) + "\n")

    return result


def ucb(task_no, instance, arms, randomSeed, horizon, epsilon = 0.02, scale = 2, threshold = 0, highs = 0):
    """
    :param arms: The actual means of the bandit arms
    :param randomSeed: The random seed to generate pseudo random results
    :param horizon: The number of pulls the bandit should make
    :param epsilon: Dummy Parameter
    :return: String of form "algorithm, random seed, epsilon, horizon, REG"
    """
    np.random.seed(randomSeed)
    num_arms = len(arms) # number of arms
    REW = 0
    rewards = np.zeros(num_arms)
    count = np.zeros(num_arms)
    ucbs = np.zeros(num_arms)
    for time in range(num_arms):
        current_reward = np.random.binomial(1, arms[time])
        count[time] += 1
        rewards[time] += current_reward
        REW += current_reward
    for time in range(num_arms, horizon):
        for i in range(num_arms):
            ucbs[i] = rewards[i] + np.sqrt(float(scale) * np.log(time)/count[i])

        arm = np.random.choice(np.where(ucbs == ucbs.max())[0])
        current_reward = np.random.binomial(1, arms[arm])
        count[arm] += 1
        rewards[arm] += (current_reward - rewards[arm]) / count[arm]
        REW += current_reward

    reg = horizon * max(arms) - REW

    result = str(instance + ", " + "ucb-t" + str(task_no) + ", " + str(randomSeed) + ", " + str(epsilon) + ", " \
            + str(scale) + ", " + str(threshold) + ", " + str(horizon) + ", " + str(reg) + ", " + str(highs) + "\n")

    return result


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

def get_kl_ucb(reward, count, time, c=3):
    tolerance = 0.005
    target = (np.log(time) + c*np.log(np.log(time)))/count
    low, high = reward, 1

    while high-low>=tolerance:
        mid = low + (high - low)/2
        res = KL(reward, mid)
        if target > res and target-res <= tolerance:
            return mid
        elif res > target:
            high = mid
        else:
            low = mid
    return low

def kl_ucb(instance, arms, randomSeed, horizon, epsilon = 0.02, scale = 2, threshold = 0, highs = 0):
    """
    :param arms: The actual means of the bandit arms
    :param randomSeed: The random seed to generate pseudo random results
    :param horizon: The number of pulls the bandit should make
    :param epsilon: Dummy Parameter
    :return: String of form "algorithm, random seed, epsilon, horizon, REG"
    """
    np.random.seed(randomSeed)
    num_arms = len(arms) # number of arms
    REW = 0
    rewards = np.zeros(num_arms)
    count = np.zeros(num_arms)
    kl_ucbs = np.zeros(num_arms)
    for time in range(num_arms):
        current_reward = np.random.binomial(1, arms[time])
        count[time] += 1
        rewards[time] += current_reward
        REW += current_reward


    for time in range(num_arms, horizon):
        for i in range(num_arms):
            kl_ucbs[i] = get_kl_ucb(rewards[i], count[i], time)
        arm = np.argmax(kl_ucbs)
        current_reward = np.random.binomial(1, arms[arm])
        count[arm] += 1
        rewards[arm] += (current_reward - rewards[arm]) / count[arm]
        REW += current_reward

    reg = horizon * max(arms) - REW

    result = str(instance + ", " + "kl-ucb-t1, " + str(randomSeed) + ", " + str(epsilon) + ", " \
            + str(scale) + ", " + str(threshold) + ", " + str(horizon) + ", " + str(reg) + ", " + str(highs) + "\n")

    return result


def thompson_sampling(instance, arms, randomSeed, horizon, file, epsilon = 0.02, scale = 2, threshold = 0, highs = 0):
    """
    :param arms: The actual means of the bandit arms
    :param randomSeed: The random seed to generate pseudo random results
    :param horizon: The number of pulls the bandit should make
    :param epsilon: Dummy Parameter
    :return: String of form "algorithm, random seed, epsilon, horizon, REG"
    """
    np.random.seed(randomSeed)
    num_arms = len(arms) # number of arms
    REW = 0
    values = np.zeros(num_arms)
    count = np.zeros(num_arms)

    for time in range(horizon):
        arm = 0
        max_sampled = -1
        for i in range(num_arms):
            s = values[i]*count[i]
            f = count[i]-s
            sampled = np.random.beta(s+1, f+1)
            if(sampled>max_sampled):
                arm = i
                max_sampled = sampled

        current_reward = np.random.binomial(1, arms[arm])
        count[arm] += 1
        values[arm] += (current_reward - values[arm]) / count[arm]
        REW += current_reward

    reg = horizon * max(arms) - REW

    result = str(instance + ", " + "thompson-sampling-t1, " + str(randomSeed) + ", " + str(epsilon) + ", " \
            + str(scale) + ", " + str(threshold) + ", " + str(horizon) + ", " + str(reg) + ", " + str(highs) + "\n")

    return result


n = len(sys.argv)

# default values to the bandit - random
instance_path, algorithm, randomSeed, epsilon, scale, threshold, horizon = "../instances/instances-task1/i-1.txt", "kl-ucb-t1", 0, 0.02, 2, 0, 400

for i in range(1, n):  # take command line arguments
    if sys.argv[i] == "--instance":
        i += 1
        instance_path = sys.argv[i]
    elif sys.argv[i] == "--algorithm":
        i += 1
        algorithm = sys.argv[i]
    elif sys.argv[i] == "--randomSeed":
        i += 1
        randomSeed = int(sys.argv[i])
    elif sys.argv[i] == "--epsilon":
        i += 1
        epsilon = float(sys.argv[i])
    elif sys.argv[i] == "--scale":
        i += 1
        scale = sys.argv[i]
    elif sys.argv[i] == "--threshold":
        i += 1
        threshold = sys.argv[i]
    elif sys.argv[i] == "--horizon":
        i += 1
        horizon = int(sys.argv[i])

with open(instance_path) as f: # read true means of the bandit instance
    arms = [line.strip() for line in f]
    arms = list(map(float, arms))
    #print(arms)



# call the appropriate function based on command line arguments
if algorithm == "epsilon-greedy-t1":
    result = epsilon_greedy(instance_path, arms, randomSeed, horizon, epsilon, scale, threshold, 0)
elif algorithm == "ucb-t1":
    task_no = 1
    result = ucb(task_no, instance_path, arms, randomSeed, horizon, epsilon, scale, threshold, 0)
elif algorithm == "ucb-t2":
    task_no = 2
    result = ucb(task_no, instance_path, arms, randomSeed, horizon, epsilon, scale, threshold, 0)
elif algorithm == "kl-ucb-t1":
    result = kl_ucb(instance_path, arms, randomSeed, horizon, epsilon, scale, threshold, 0)
elif algorithm == "thompson-sampling-t1":
    result = thompson_sampling(instance_path, arms, randomSeed, horizon, epsilon, scale, threshold, 0)


print(result) # print result in the right format