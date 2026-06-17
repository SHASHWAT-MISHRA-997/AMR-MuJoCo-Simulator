# RL Training Pipeline

This project now includes two reinforcement-learning training scaffolds:

- `dock`: single-AMR docking and navigation PPO environment
- `fleet`: dual-AMR warehouse PPO scaffold for AMR-A + AMR-B coordination experiments

## Train Single-AMR Dock Policy

```bash
python3 -m mujoco_amr_sim.train_rl --env dock --timesteps 300000
```

## Train Dual-AMR Fleet Policy

```bash
python3 -m mujoco_amr_sim.train_rl --env fleet --timesteps 500000
```

## What This Gives You

- reproducible PPO training entry point
- TensorBoard logs
- saved `.zip` policies
- concrete multi-robot state/action scaffold to extend further

## Current Limits

- this is a training scaffold, not a finished trained production policy
- the `fleet` environment is intentionally simplified so it can be extended for richer rewards, task graphs, and traffic rules
- photoreal or scanned-asset digital twin training is not part of this scaffold
