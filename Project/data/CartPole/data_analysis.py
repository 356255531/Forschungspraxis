import pickle
import matplotlib
from matplotlib import pyplot as plt


def read_data():
    data = {}
    for data_type in ["time_history", "total_reward", ]:
        data[data_type] = {}
        for algo in ["GGQ", "RGGQ", "OSKQ"]:
            data[data_type][algo] = {}
            for learning_rate in [0.03, 0.1]:
                data[data_type][algo][learning_rate] = {}
                for eligibility_factor in [0, 0.1]:
                    data[data_type][algo][learning_rate][eligibility_factor] = {}

                    if "RGGQ" == algo:
                        for regularize_factor in [0.00003, 0.0001, 0.0003]:
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

                    if "OSKQ" == algo:
                        for mu_2 in [0.08, 0.16]:
                            with open(
                                    data_type + "_" + algo + "-" + str(learning_rate) + "-" +
                                    str(eligibility_factor) + "-" + str(0.04) + "-" + str(mu_2),
                                    'rb'
                            ) as f:
                                data[data_type][algo][learning_rate][eligibility_factor][mu_2] = pickle.load(f)

    data["sparsity"] = {}
    data["sparsity"]["RGGQ"] = {}
    for learning_rate in [0.03, 0.1]:
        data["sparsity"]["RGGQ"][learning_rate] = {}
        for eligibility_factor in [0, 0.1]:
            data["sparsity"]["RGGQ"][learning_rate][eligibility_factor] = {}
            for regularize_factor in [0.00003, 0.0001, 0.0003]:
                with open(
                    "sparsity_RGGQ" + "-" + str(learning_rate) + "-" +
                    str(eligibility_factor) + "-" + str(regularize_factor),
                    'rb'
                ) as f:
                    data["sparsity"]["RGGQ"][learning_rate][eligibility_factor][regularize_factor] = pickle.load(f)
    return data


def save_img(data):
    font = {'family': 'normal',
            'size': 10}
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(16, 18))
    for eligibility_factor_idx, eligibility_factor in enumerate([0, 0.1]):
        for learning_rate_idx, learning_rate in enumerate([0.03, 0.1]):
            ax = fig.add_subplot(
                2, 2, eligibility_factor_idx * 2 + learning_rate_idx + 1)
            ax.set_xlabel("Episodes: learning_rate=" + str(learning_rate) + " eligibility_factor=" + str(eligibility_factor))
            ax.set_ylabel('Sum of reward')

            line_1, = plt.plot(data["total_reward"]["GGQ"][learning_rate][eligibility_factor])
            line_2, = plt.plot(data["total_reward"]["RGGQ"][learning_rate][eligibility_factor][0.00003])
            line_3, = plt.plot(data["total_reward"]["RGGQ"][learning_rate][eligibility_factor][0.0001])
            line_4, = plt.plot(data["total_reward"]["RGGQ"][learning_rate][eligibility_factor][0.0003])
            line_5, = plt.plot(data["total_reward"]["OSKQ"][learning_rate][eligibility_factor][0.08])
            line_6, = plt.plot(data["total_reward"]["OSKQ"][learning_rate][eligibility_factor][0.16])

    fig.legend(
        (line_1, line_2, line_3, line_4, line_5, line_6),
        (
            "GQ(lambda)",
            "RGGQ(lambda)(regularize_factor=0.00003)",
            "RGGQ(lambda)(regularize_factor=0.0001)",
            "RGGQ(lambda)(regularize_factor=0.0003)",
            "OSKQ(mu_2=0.08)",
            "OSKQ(mu_2=0.16)",
        ),
        loc=8)

    fig.savefig("Total Reward.jpg")

    fig = plt.figure(figsize=(16, 18))
    for eligibility_factor_idx, eligibility_factor in enumerate([0, 0.1]):
        for learning_rate_idx, learning_rate in enumerate([0.03, 0.1]):
            ax = fig.add_subplot(
                2, 2, eligibility_factor_idx * 2 + learning_rate_idx + 1)
            ax.set_xlabel("Episodes: learning_rate=" + str(learning_rate) + " eligibility_factor=" + str(eligibility_factor))
            ax.set_ylabel('Sum of consumed time')

            line_1, = plt.plot(data["time_history"]["GGQ"][learning_rate][eligibility_factor])
            line_2, = plt.plot(data["time_history"]["RGGQ"][learning_rate][eligibility_factor][0.00003])
            line_3, = plt.plot(data["time_history"]["RGGQ"][learning_rate][eligibility_factor][0.0001])
            line_4, = plt.plot(data["time_history"]["RGGQ"][learning_rate][eligibility_factor][0.0003])
            line_5, = plt.plot(data["time_history"]["OSKQ"][learning_rate][eligibility_factor][0.08])
            line_6, = plt.plot(data["time_history"]["OSKQ"][learning_rate][eligibility_factor][0.16])

    fig.legend(
        (line_1, line_2, line_3, line_4, line_5, line_6,),
        (
            "GQ(lambda)",
            "RGGQ(lambda)(regularize_factor=0.00003)",
            "RGGQ(lambda)(regularize_factor=0.0001)",
            "RGGQ(lambda)(regularize_factor=0.0003)",
            "OSKQ(mu_2=0.08)",
            "OSKQ(mu_2=0.16)",
        ),
        loc=8)
    fig.savefig("Total time consumed.jpg")

    fig = plt.figure(figsize=(12, 12))
    for eligibility_factor_idx, eligibility_factor in enumerate([0, 0.1]):
        for learning_rate_idx, learning_rate in enumerate([0.03, 0.1]):
            ax = fig.add_subplot(
                2, 2, eligibility_factor_idx * 2 + learning_rate_idx + 1)
            ax.set_xlabel("Episodes: learning_rate=" + str(learning_rate) + " eligibility_factor=" + str(eligibility_factor))
            ax.set_ylabel('Sparsity')

            line_1, = plt.plot(data["sparsity"]["RGGQ"][learning_rate][eligibility_factor][0.00003])
            line_2, = plt.plot(data["sparsity"]["RGGQ"][learning_rate][eligibility_factor][0.0001])
            line_3, = plt.plot(data["sparsity"]["RGGQ"][learning_rate][eligibility_factor][0.0003])

    fig.legend(
        (line_1, line_2, line_3),
        (
            "RGGQ(lambda)(regularize_factor=0.00003)",
            "RGGQ(lambda)(regularize_factor=0.0001)",
            "RGGQ(lambda)(regularize_factor=0.0003)",
        ),
        loc=8)
    fig.savefig("Sparsity.jpg")


def data_denoise(data, ave_step=5):
    if isinstance(data, dict):
        new_data = {}
        for key in data.keys():
            new_data[key] = data_denoise(data[key], ave_step)
    else:
        new_data = []
        for ele in range(len(data) - ave_step):
            new_data.append(sum([data[ele + i] for i in range(ave_step)]) / ave_step)
    return new_data


def main():
    data = read_data()
    data = data_denoise(data, ave_step=50)
    save_img(data)


if __name__ == '__main__':
    main()
