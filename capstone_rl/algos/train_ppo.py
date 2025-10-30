import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from capstone_rl.envs.stub_srb import StubSRBEnv

def make_env(stub=False):
    if stub:
        return StubSRBEnv()
    else:
        import gymnasium as gym
        task = "srb/waypoint_navigation-moon-wheeled-state-v0"
        return gym.make(task)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stub", action="store_true", help="Use stub environment")
    parser.add_argument("--steps", type=int, default=10000, help="Training timesteps")
    args = parser.parse_args()

    # where SB3 will write CSV logs
    log_dir = "capstone_rl/logs/sb3"
    os.makedirs(log_dir, exist_ok=True)

    env = make_env(stub=args.stub)
    #env = make_env(stub=False)
    model = PPO("MlpPolicy", env, verbose=1)

    # configure SB3 to write CSV logs into log_dir/progress.csv
    new_logger = configure(log_dir, ["csv"])   # you can add "tensorboard" too
    model.set_logger(new_logger)

    model.learn(total_timesteps=args.steps)
    model.save("ppo_stub_model")

if __name__ == "__main__":
    main()