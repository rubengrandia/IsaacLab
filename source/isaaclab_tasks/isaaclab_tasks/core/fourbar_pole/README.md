# Fourbar Pole Swing-up

A variant of the classical cartpole where a parallel **four-bar linkage** drives the
"cart" (the `coupler`) instead of a linear rail. A pole is mounted on the coupler and
starts hanging down; the task is to swing it up and balance it upright.

The four-bar forms a closed kinematic loop (`ground -> crank -> coupler -> rocker -> ground`).
The `rocker_to_ground` joint is marked `excludeFromArticulation` in the robot USD so the
solver treats it as a loop-closure constraint. Only the **kamino** Newton backend is wired
up for now.

## Registered task

| Task ID | Workflow | Entry point |
| --- | --- | --- |
| `Isaac-Fourbar-Pole-Swingup` | Manager-based | `fourbar_pole_manager_env_cfg:FourbarPoleSwingupEnvCfg` |

The kamino backend is the default, so the `physics=newton_kamino` token below is optional
but kept explicit for clarity.

## Commands

Run all commands from the Isaac Lab repository root.

### Zero-agent (sanity check)

Steps the environment with zero actions to validate that the scene loads and simulates.

```bash
./isaaclab.sh -p scripts/environments/zero_agent.py \
    --task Isaac-Fourbar-Pole-Swingup \
    --num_envs 16 \
    physics=newton_kamino \
    --viz newton
```

Add `--headless` to run without the GUI.

### Train (RSL-RL)

```bash
./isaaclab.sh train --rl_library rsl_rl \
    --task Isaac-Fourbar-Pole-Swingup \
    --headless \
    physics=newton_kamino
```

Useful overrides: `--num_envs <N>`, `--max_iterations <N>`, `--seed <N>`. Checkpoints and
logs are written under `logs/rsl_rl/fourbar_pole/`.

### Play (visualize a trained policy)

```bash
./isaaclab.sh play --rl_library rsl_rl \
    --task Isaac-Fourbar-Pole-Swingup \
    --num_envs 16 \
    physics=newton_kamino \
    --viz newton
```

By default this loads the most recent checkpoint from `logs/rsl_rl/fourbar_pole/`. Use
`--load_run <run_dir>` and/or `--checkpoint <file>` to select a specific policy.
