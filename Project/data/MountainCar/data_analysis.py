import pickle
import matplotlib
from matplotlib import pyplot as plt
import numpy.ma as ma
import numpy as np
from pykalman import KalmanFilter


def read_data():
    data = {}
    for data_type in ["time_history", "total_reward"]:
        data[data_type] = {}
        for algo in ["GGQ", "RGGQ", "OSKQ"]:
            data[data_type][algo] = {}
            for learning_rate in [0.003, 0.01, 0.03, 0.1]:
                data[data_type][algo][learning_rate] = {}
                for eligibility_factor in [0.3, 0.6, 0.9]:
                    data[data_type][algo][learning_rate][eligibility_factor] = {}
                    if "OSKQ" == algo:
                        for mu_1 in [0.04]:
                            data[data_type][algo][learning_rate][eligibility_factor][mu_1] = {}
                            for mu_2 in [0.04, 0.08]:
                                with open(
                                        data_type + "_" + algo + "-" + str(learning_rate) +
                                        "-" + str(eligibility_factor) + "-" + str(mu_1) + "-" + str(mu_2),
                                        'rb'
                                ) as f:
                                    data[data_type][algo][learning_rate][eligibility_factor][mu_1][mu_2] = pickle.load(f)

                    if "RGGQ" == algo:
                        for regularize_factor in [0.001, 0.003, 0.01]:
                            with open(
                                    data_type + "_" + algo + "-" + str(learning_rate) + "-" +
                                    str(eligibility_factor) + "-" + str(regularize_factor),
                                    'rb'
                            ) as f:
                                data[data_type][algo][learning_rate][eligibility_factor][regularize_factor] = pickle.load(f)

                    if "GGQ" == algo:
                        with open(
                                data_type + "_" + algo + "-" + str(learning_rate) + "-" + str(eligibility_factor),
                                'rb'
                        ) as f:
                            data[data_type][algo][learning_rate][eligibility_factor] = pickle.load(f)
    return data


def save_img(data):
    font = {'family': 'normal',
            'size': 7}
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(16, 16))
    for learning_rate_idx, learning_rate in enumerate([0.003, 0.01, 0.03, 0.1]):
        for eligibility_factor_idx, eligibility_factor in enumerate([0.3, 0.6, 0.9]):
            ax = fig.add_subplot(
                4, 3, learning_rate_idx * 3 + eligibility_factor_idx + 1)
            ax.set_xlabel("Episodes: learning_rate=" + str(learning_rate) + " eligibility_factor=" + str(eligibility_factor))
            ax.set_ylabel('Sum of reward')

            line_1, = plt.plot(data["total_reward"]["GGQ"][learning_rate][eligibility_factor][:550])
            line_2, = plt.plot(data["total_reward"]["RGGQ"][learning_rate][eligibility_factor][0.001][:550])
            line_3, = plt.plot(data["total_reward"]["RGGQ"][learning_rate][eligibility_factor][0.003][:550])
            line_4, = plt.plot(data["total_reward"]["RGGQ"][learning_rate][eligibility_factor][0.01][:550])
            # line_5, = plt.plot(data["total_reward"]["OSKQ"][learning_rate][eligibility_factor][0.04])
            # line_6, = plt.plot(data["total_reward"]["OSKQ"][learning_rate][eligibility_factor][0.08])
    fig.legend(
        (line_1, line_2, line_3, line_4),
        (
            "GQ(lambda)", "RGGQ(lambda)(regularize_factor=0.001)",
            "RGGQ(lambda)(regularize_factor=0.003)", "RGGQ(lambda)(regularize_factor=0.01)",
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
    for learning_rate_idx, learning_rate in enumerate([0.003, 0.01, 0.03, 0.1]):
        for eligibility_factor_idx, eligibility_factor in enumerate([0.3, 0.6, 0.9]):
            ax = fig.add_subplot(
                4, 3, learning_rate_idx * 3 + eligibility_factor_idx + 1)
            ax.set_xlabel("Episodes: learning_rate=" + str(learning_rate) + " eligibility_factor=" + str(eligibility_factor))
            ax.set_ylabel('Sum of reward')

            line_1, = plt.plot(data["time_history"]["GGQ"][learning_rate][eligibility_factor][:550])
            line_2, = plt.plot(data["time_history"]["RGGQ"][learning_rate][eligibility_factor][0.001][:550])
            line_3, = plt.plot(data["time_history"]["RGGQ"][learning_rate][eligibility_factor][0.003][:550])
            line_4, = plt.plot(data["time_history"]["RGGQ"][learning_rate][eligibility_factor][0.01][:550])
            line_5, = plt.plot(data["time_history"]["OSKQ"][learning_rate][eligibility_factor][0.04][0.04][:550])
            line_6, = plt.plot(data["time_history"]["OSKQ"][learning_rate][eligibility_factor][0.04][0.08][:550])

    fig.legend(
        (line_1, line_2, line_3, line_4, line_5, line_6),
        (
            "GQ(lambda)", "RGGQ(lambda)(regularize_factor=0.001)",
            "RGGQ(lambda)(regularize_factor=0.003)", "RGGQ(lambda)(regularize_factor=0.01)",
            "OKS-Q(mu2=0.04)", "OKS-Q(mu2=0.08)"),
        loc=8)
    fig.savefig("Total time consumed.jpg")


def data_denoise(data, ave_step=5):
    if isinstance(data, dict):
        new_data = {}
        for key in data.keys():
            new_data[key] = data_denoise(data[key], ave_step)
    else:
        new_data = []
        for ele_idx in range(len(data) - ave_step):
            new_data.append(sum([data[ele_idx + i] for i in range(ave_step)]) / (ave_step * 1.0))
    return new_data


def main():
    data = read_data()
    data = data_denoise(data, ave_step=20)
    save_img(data)


if __name__ == '__main__':
    main()
