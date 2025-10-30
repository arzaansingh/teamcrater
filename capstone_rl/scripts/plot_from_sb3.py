import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="Path to SB3 progress.csv")
    p.add_argument("--out", default=None, help="Optional PNG path to save")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    # SB3 column names
    x = df["time/total_timesteps"]
    y = df["rollout/ep_rew_mean"]

    print(df[["time/total_timesteps", "rollout/ep_rew_mean"]].head())

    plt.figure(figsize=(8,5))
    plt.plot(x, y, marker="o", linewidth=2, label="Episode Reward Mean")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward Mean")
    plt.title("PPO Training Curve (Stub Environment)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    if args.out:
        plt.tight_layout(); plt.savefig(args.out)
    else:
        plt.show()

if __name__ == "__main__":
    main()