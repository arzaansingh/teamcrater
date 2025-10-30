import argparse
import csv
import matplotlib.pyplot as plt

def read_csv_log(csv_path):
    """
    Read a CSV log file with columns like 'timesteps' and 'reward' (or 'ep_rew_mean').
    Returns two lists: timesteps_list, rewards_list.
    """
    timesteps = []
    rewards = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # adjust keys depending on your log file
            if "timesteps" in row:
                timesteps.append(int(row["timesteps"]))
            elif "total_timesteps" in row:
                timesteps.append(int(row["total_timesteps"]))
            else:
                # fallback: if your log has a column "step" or similar, adjust here
                timesteps.append(len(timesteps))
            # reward or ep_rew_mean
            if "reward" in row:
                rewards.append(float(row["reward"]))
            elif "ep_rew_mean" in row:
                rewards.append(float(row["ep_rew_mean"]))
            else:
                # fallback: try another key
                # assuming your log has some reward column
                k = next((k for k in row if "rew" in k.lower()), None)
                if k is not None:
                    rewards.append(float(row[k]))
                else:
                    rewards.append(0.0)
    return timesteps, rewards

def plot_curve(csv_path, output_path=None):
    timesteps, rewards = read_csv_log(csv_path)
    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, rewards, label="Episode Return")
    plt.xlabel("Timesteps")
    plt.ylabel("Return")
    plt.title("Learning Curve")
    plt.grid(True)
    plt.legend()
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=str, help="Path to CSV log file")
    parser.add_argument("--out", type=str, default=None, help="Path to save plot (PNG)")
    args = parser.parse_args()
    plot_curve(args.csv, output_path=args.out)
