import pickle
import matplotlib
from matplotlib import pyplot as plt


def save_img(data):
    font = {'family': 'normal',
            'size': 7}
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(16, 16))
    for learning_rate_idx, learning_rate in enumerate([0.001, 0.003, 0.01, 0.03, 0.1]):
        for eligibility_factor_idx, eligibility_factor in enumerate([0.2, 0.4, 0.6, 0.8]):
            ax = fig.add_subplot(
                5, 4, learning_rate_idx * 4 + eligibility_factor_idx + 1)
            ax.set_xlabel("Episodes: learning_rate=" + str(learning_rate) + " eligibility_factor=" + str(eligibility_factor))
            ax.set_ylabel('Sum of reward')

            line_1, = plt.plot(data["total_reward"]["GGQ"][learning_rate][eligibility_factor])
            line_2, = plt.plot(data["total_reward"]["RGGQ"][learning_rate][eligibility_factor])
            line_3, = plt.plot(data["total_reward"]["OSK-Q"][learning_rate][eligibility_factor][0.04])
            line_4, = plt.plot(data["total_reward"]["OSK-Q"][learning_rate][eligibility_factor][0.08])

    fig.legend((line_1, line_2, line_3, line_4),
               ("GQ(lambda)", "RGGQ(lambda)", "OKS-Q(mu2=0.04)", "OKS-Q(mu2=0.08)"),
               loc=8)
    fig.savefig("Total Reward.jpg")

    fig = plt.figure(figsize=(16, 16))
    for learning_rate_idx, learning_rate in enumerate([0.001, 0.003, 0.01, 0.03, 0.1]):
        for eligibility_factor_idx, eligibility_factor in enumerate([0.2, 0.4, 0.6, 0.8]):
            ax = fig.add_subplot(
                5, 4, learning_rate_idx * 4 + eligibility_factor_idx + 1)
            ax.set_xlabel("Episodes: learning_rate=" + str(learning_rate) + " eligibility_factor=" + str(eligibility_factor))
            ax.set_ylabel('Sum of reward')

            line_1, = plt.plot(data["time_history"]["GGQ"][learning_rate][eligibility_factor])
            line_2, = plt.plot(data["time_history"]["RGGQ"][learning_rate][eligibility_factor])
            line_3, = plt.plot(data["time_history"]["OSK-Q"][learning_rate][eligibility_factor][0.04])
            line_4, = plt.plot(data["time_history"]["OSK-Q"][learning_rate][eligibility_factor][0.08])
    fig.legend((line_1, line_2, line_3, line_4),
               ("GQ(lambda)", "RGGQ(lambda)", "OKS-Q(mu2=0.04)", "OKS-Q(mu2=0.08)"),
               loc=8)
    fig.savefig("Total time consumed.jpg")


def main():
    data = {}
    for data_type in ["time_history", "total_reward"]:
        data[data_type] = {}
        for algo in ["GGQ", "RGGQ"]:
            data[data_type][algo] = {}
            for learning_rate in [0.001, 0.003, 0.01, 0.03, 0.1]:
                data[data_type][algo][learning_rate] = {}
                for eligibility_factor in [0.2, 0.4, 0.6, 0.8]:
                    with open(
                            data_type + "_" + algo + "-" + str(learning_rate) + "-" + str(eligibility_factor),
                            'rb'
                    ) as f:
                        data[data_type][algo][learning_rate][eligibility_factor] = pickle.load(f)[:200]

    for data_type in ["time_history", "total_reward"]:
        data[data_type]["OSK-Q"] = {}
        for learning_rate in [0.001, 0.003, 0.01, 0.03, 0.1]:
            data[data_type]["OSK-Q"][learning_rate] = {}
            for eligibility_factor in [0.2, 0.4, 0.6, 0.8]:
                data[data_type]["OSK-Q"][learning_rate][eligibility_factor] = {}
                for mu_2 in [0.04, 0.08]:
                    with open(
                            data_type + "_OSKQ-" + str(learning_rate) + "-" + str(eligibility_factor) + "-" + str(mu_2),
                            'rb'
                    ) as f:
                        data[data_type]["OSK-Q"][learning_rate][eligibility_factor][mu_2] = pickle.load(f)[:200]

    save_img(data)


if __name__ == '__main__':
    main()