import argparse
from pathlib import Path


def _build_env(env_name: str):
    from .rl_env import MultiRobotWarehouseEnv, MujocoDockNavEnv

    normalized = (env_name or "dock").strip().lower()
    if normalized == "fleet":
        return MultiRobotWarehouseEnv(), "mujoco_amr_fleet_rl"
    return MujocoDockNavEnv(), "mujoco_amr_dock_rl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO policies for MuJoCo AMR simulation tasks.")
    parser.add_argument("--env", choices=["dock", "fleet"], default="dock", help="Training environment profile")
    parser.add_argument("--timesteps", type=int, default=300000, help="Total PPO timesteps")
    parser.add_argument("--output-dir", default="models", help="Model output directory")
    parser.add_argument("--tensorboard-dir", default="runs", help="TensorBoard output directory")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise RuntimeError(
            "stable-baselines3 is required for RL training. Install gymnasium and stable-baselines3 in WSL first."
        ) from exc

    base_env, log_name = _build_env(args.env)
    check_env(base_env)
    env = Monitor(base_env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        tensorboard_log=str(Path(args.tensorboard_dir) / log_name),
    )
    model.learn(total_timesteps=max(10000, int(args.timesteps)))

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / f"{log_name}_ppo"
    model.save(model_path)
    print(f"Saved policy to {model_path}.zip")


if __name__ == "__main__":
    main()
