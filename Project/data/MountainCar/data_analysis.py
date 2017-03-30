import pickle
import matplotlib
from matplotlib import pyplot as plt
import numpy as np


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

            line_1, = plt.plot(data["total_reward"]["GGQ"][learning_rate][eligibility_factor][:200])
            line_2, = plt.plot(data["total_reward"]["RGGQ"][learning_rate][eligibility_factor][0.001][:200])
            line_3, = plt.plot(data["total_reward"]["RGGQ"][learning_rate][eligibility_factor][0.003][:200])
            line_4, = plt.plot(data["total_reward"]["RGGQ"][learning_rate][eligibility_factor][0.01][:200])
            line_5, = plt.plot(data["total_reward"]["RGGQ"][learning_rate][eligibility_factor][0.03][:200])
            # line_6, = plt.plot(data["total_reward"]["OSKQ"][learning_rate][eligibility_factor][0.04])
            # line_7, = plt.plot(data["total_reward"]["OSKQ"][learning_rate][eligibility_factor][0.08])
    fig.legend(
        (line_1, line_2, line_3, line_4, line_5),
        (
            "GQ(lambda)", "RGGQ(lambda)(regularize_factor=0.001)",
            "RGGQ(lambda)(regularize_factor=0.003)", "RGGQ(lambda)(regularize_factor=0.01)",
            "RGGQ(lambda)(regularize_factor=0.03)"
        ),
        loc=8)

    # fig.legend(
    #     (line_1, line_2, line_3, line_4, line_5, line_6, line_7),
    #     (
    #         "GQ(lambda)", "RGGQ(lambda)(regularize_factor=0.001)",
    #         "RGGQ(lambda)(regularize_factor=0.003)", "RGGQ(lambda)(regularize_factor=0.01)",
    #         "RGGQ(lambda)(regularize_factor=0.03)",
    #         "OKS-Q(mu2=0.04)", "OKS-Q(mu2=0.08)"),
    #     loc=8)
    fig.savefig("Total Reward.jpg")

    fig = plt.figure(figsize=(16, 16))
    for learning_rate_idx, learning_rate in enumerate([0.001, 0.003, 0.01, 0.03, 0.1]):
        for eligibility_factor_idx, eligibility_factor in enumerate([0.2, 0.4, 0.6, 0.8]):
            ax = fig.add_subplot(
                5, 4, learning_rate_idx * 4 + eligibility_factor_idx + 1)
            ax.set_xlabel("Episodes: learning_rate=" + str(learning_rate) + " eligibility_factor=" + str(eligibility_factor))
            ax.set_ylabel('Sum of reward')

            line_1, = plt.plot(data["time_history"]["GGQ"][learning_rate][eligibility_factor][:200])
            line_2, = plt.plot(data["time_history"]["RGGQ"][learning_rate][eligibility_factor][0.001][:200])
            line_3, = plt.plot(data["time_history"]["RGGQ"][learning_rate][eligibility_factor][0.003][:200])
            line_4, = plt.plot(data["time_history"]["RGGQ"][learning_rate][eligibility_factor][0.01][:200])
            line_5, = plt.plot(data["time_history"]["RGGQ"][learning_rate][eligibility_factor][0.03][:200])
            line_6, = plt.plot(data["time_history"]["OSKQ"][learning_rate][eligibility_factor][0.04])
            line_7, = plt.plot(data["time_history"]["OSKQ"][learning_rate][eligibility_factor][0.08])

    fig.legend(
        (line_1, line_2, line_3, line_4, line_5, line_6, line_7),
        (
            "GQ(lambda)", "RGGQ(lambda)(regularize_factor=0.001)",
            "RGGQ(lambda)(regularize_factor=0.003)", "RGGQ(lambda)(regularize_factor=0.01)",
            "RGGQ(lambda)(regularize_factor=0.03)",
            "OKS-Q(mu2=0.04)", "OKS-Q(mu2=0.08)"),
        loc=8)
    fig.savefig("Total time consumed.jpg")


def data_average(data, ave_step=5):
    if isinstance(data, dict):
        new_data = {}
        for key in data.keys():
            new_data[key] = data_average(data[key], ave_step)
    else:
        new_data = list(np.convolve(np.array(data), [1 / (ave_step * 1.0) for time in range(ave_step)], 'same'))
        for idx_num in range(ave_step):
            new_data[idx_num] = sum(data[0:idx_num + 1]) / float(len(data[0:idx_num + 1]))
            new_data[len(new_data) - idx_num - 1] = sum(data[len(new_data) - idx_num - 1:len(new_data)]) / float(len(data[len(new_data) - idx_num - 1:len(new_data)]))
    return new_data


def main():
    data = {}
    for data_type in ["time_history", "total_reward"]:
        data[data_type] = {}
        for algo in ["GGQ", "RGGQ", "OSKQ"]:
            data[data_type][algo] = {}
            for learning_rate in [0.001, 0.003, 0.01, 0.03, 0.1]:
                data[data_type][algo][learning_rate] = {}
                for eligibility_factor in [0.2, 0.4, 0.6, 0.8]:
                    data[data_type][algo][learning_rate][eligibility_factor] = {}
                    if "OSKQ" == algo:
                        for mu_2 in [0.04, 0.08]:
                            with open(
                                    data_type + "_" + algo + "-" + str(learning_rate) + "-" + str(eligibility_factor) + "-" + str(mu_2),
                                    'rb'
                            ) as f:
                                data[data_type][algo][learning_rate][eligibility_factor][mu_2] = pickle.load(f)[:230]

                    if "RGGQ" == algo:
                        for regularize_factor in [0.001, 0.003, 0.01, 0.03]:
                            with open(
                                    data_type + "_" + algo + "-" + str(learning_rate) + "-" + str(eligibility_factor) + "-" + str(regularize_factor),
                                    'rb'
                            ) as f:
                                data[data_type][algo][learning_rate][eligibility_factor][regularize_factor] = pickle.load(f)[:230]

                    if "GGQ" == algo:
                        with open(
                                data_type + "_" + algo + "-" + str(learning_rate) + "-" + str(eligibility_factor),
                                'rb'
                        ) as f:
                            data[data_type][algo][learning_rate][eligibility_factor] = pickle.load(f)[:230]

    data = data_average(data, ave_step=20)
    save_img(data)


if __name__ == '__main__':
    main()
