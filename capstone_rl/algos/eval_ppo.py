import argparse
from stable_baselines3 import PPO
# Import your stub environment
from capstone_rl.envs.stub_srb import StubSRBEnv

def make_env(stub=False):
    if stub:
        return StubSRBEnv()
    else:
        import gymnasium as gym
        task = "srb/waypoint_navigation-moon-wheeled-state-v0"
        return gym.make(task)

def evaluate(model_path, stub=True, num_episodes=5):
    model = PPO.load(model_path)
    env = make_env(stub=stub)
    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {ep} — Return: {total_reward:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ppo_stub_model")
    parser.add_argument("--stub", action="store_true", help="Use stub environment (default)")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    evaluate(args.model, stub=args.stub, num_episodes=args.episodes)