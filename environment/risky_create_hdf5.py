import argparse
from pathlib import Path

import d4rl
import gym
import h5py
import numpy as np
import yaml

from environment.risky_wrappers import make_risky_env


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def wrap_env_from_config(base_env, cfg):
    wrapper_cfg = cfg["wrapper"]
    return make_risky_env(
        base_env,
        max_vel=wrapper_cfg.get("max_vel", 3.0),
        prob_vel_penal=wrapper_cfg.get("prob_vel_penal", 0.05),
        cost_vel=wrapper_cfg.get("cost_vel", -20.0),
        prob_pose_penal=wrapper_cfg.get("prob_pose_penal", 0.1),
        cost_pose=wrapper_cfg.get("cost_pose", -50.0),
        healthy_angle_range=wrapper_cfg.get("healthy_angle_range", [-0.1, 0.1]),
        done_if_exceed_factor=wrapper_cfg.get("done_if_exceed_factor", 2.0),
    )


def reconstruct_mujoco_state(env, observation):
    """Recover ``(qpos, qvel)`` from a D4RL MuJoCo observation.

    D4RL locomotion observations typically omit the absolute root x-position,
    so the observation dimension can be either ``nq + nv`` or ``nq + nv - 1``.
    In the latter case we insert a zero root position before calling
    ``env.set_state``.
    """
    model = env.unwrapped.model
    nq, nv = model.nq, model.nv
    obs_dim = observation.shape[0]

    if obs_dim == nq + nv:
        qpos = observation[:nq]
        qvel = observation[nq:nq + nv]
        return qpos, qvel

    if obs_dim == nq + nv - 1:
        qpos = np.zeros(nq, dtype=np.float64)
        qpos[1:] = observation[: nq - 1]
        qvel = observation[nq - 1 : nq - 1 + nv]
        return qpos, qvel

    raise ValueError(
        f"Unsupported observation/state layout: obs_dim={obs_dim}, nq={nq}, nv={nv}"
    )


def create_risky_dataset(config_path):
    cfg = load_config(config_path)
    env_name = cfg["env_name"]
    output_file = Path(cfg["output_file"])

    base_env = gym.make(env_name)
    risky_env = wrap_env_from_config(gym.make(env_name), cfg)
    dataset = d4rl.qlearning_dataset(base_env)

    observations = dataset["observations"]
    actions = dataset["actions"]
    next_observations = dataset["next_observations"]
    terminals = dataset["terminals"]

    rewards = []
    risky_states = []

    risky_env.reset()
    for obs, act, next_obs in zip(observations, actions, next_observations):
        qpos, qvel = reconstruct_mujoco_state(risky_env, obs)
        risky_env.unwrapped.set_state(qpos, qvel)
        stepped_next_obs, reward, _, info = risky_env.step(act)

        rewards.append(reward)
        risky_states.append(info.get("risky_state", False))

        if stepped_next_obs.shape != next_obs.shape:
            raise ValueError("Observed shape mismatch while rebuilding risky dataset")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_file, "w") as f:
        f.create_dataset("observations", data=observations.astype(np.float32))
        f.create_dataset("actions", data=actions.astype(np.float32))
        f.create_dataset("rewards", data=np.asarray(rewards, dtype=np.float32))
        f.create_dataset("next_observations", data=next_observations.astype(np.float32))
        f.create_dataset("terminals", data=terminals.astype(np.float32))
        f.create_dataset("risky_states", data=np.asarray(risky_states, dtype=np.bool_))



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML risky-dataset config")
    return parser.parse_args()


if __name__ == "__main__":
    create_risky_dataset(parse_args().config)
