import numpy as np


def parse_file(data, file, seed_num=5):
    retain_acc, forget_acc, test_acc, mia = [], [], [], []
    for line in file:
        if "model" in line:
            model = " ".join(line.strip().split()[1:])
            if model not in data:
                data[model] = {}
        if "alpha" in line:
            alpha = float(line.strip().split()[1])
        if "retain acc" in line:
            retain_acc.append(float(line.strip().split()[-1]))
        if "forget acc" in line:
            forget_acc.append(float(line.strip().split()[-1]))
        if "test acc" in line:
            test_acc.append(float(line.strip().split()[-1]))
        if "MIA:" in line:
            mia.append(100 * float(line.strip().split()[-1]))
        if len(retain_acc) == len(forget_acc) == len(mia) == seed_num:
            avg_retain_acc = np.mean(retain_acc)
            avg_forget_acc = np.mean(forget_acc)
            avg_test_acc = np.mean(test_acc)
            avg_mia = np.mean(mia)

            std_retain_acc = np.std(retain_acc)
            std_forget_acc = np.std(forget_acc)
            std_test_acc = np.std(test_acc)
            std_mia = np.std(mia)

            avg_gap = np.mean(
                np.abs(
                    np.array([avg_forget_acc, avg_retain_acc, avg_test_acc, avg_mia])
                    - ref
                )
            )
            tow = np.mean(
                np.abs(
                    np.array([avg_forget_acc, avg_retain_acc, avg_test_acc]) - ref[:3]
                )
            )
            avg_std = np.mean(
                np.array([std_retain_acc, std_forget_acc, std_test_acc, std_mia])
            )

            data[model][alpha] = {
                "avg forget acc": avg_forget_acc,
                "avg retain acc": avg_retain_acc,
                "avg test acc": avg_test_acc,
                "avg mia": avg_mia,
                "avg gap": avg_gap,
                "avg std": avg_std,
                "std forget acc": std_forget_acc,
                "std retain acc": std_retain_acc,
                "std test acc": std_test_acc,
                "tow": tow,
                "std mia": std_mia,
            }
            retain_acc, forget_acc, test_acc, mia = [], [], [], []
    return data


if __name__ == "__main__":
    ref = np.array([5.39, 100, 94.248, 76.264])  # sgd train
    # ref = np.array([14.71, 99.55, 85.49, 69.3])  # tiny imagenet
    # ref = np.array([6.556,100,92.838,78.362])  # adam train
    data = {}
    paths = ["logs/dual_ll.log", "logs/dual_sl.log"]

    for path in paths:
        with open(path, "r") as file:
            data = parse_file(data, file)

    for model_name, model_dict in data.items():
        print("------------------------------------")
        print("MODEL:", model_name)
        for i, key in enumerate(sorted(model_dict.keys())):
            print(
                f"alpha: {key}, UA: {model_dict[key]['avg forget acc']:.2f}+-{model_dict[key]['std forget acc']:.2f}, "
                f"RA: {model_dict[key]['avg retain acc']:.2f}+-{model_dict[key]['std retain acc']:.2f}, "
                f"TA: {model_dict[key]['avg test acc']:.2f}+-{model_dict[key]['std test acc']:.2f}, "
                f"MIA: {model_dict[key]['avg mia']:.2f}+-{model_dict[key]['std mia']:.2f}, "
                f"Avg Gap: {model_dict[key]['avg gap']:.2f}, ToW: {model_dict[key]['tow']:.2f}, Avg std: {model_dict[key]['avg std']:.2f}"
            )
