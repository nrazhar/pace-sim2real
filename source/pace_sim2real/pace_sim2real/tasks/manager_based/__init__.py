# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

import gymnasium as gym  # noqa: F401

gym.register(
    id="Isaac-Pace-Anymal-D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pace.anymal_pace_env_cfg:AnymalDPaceEnvCfg"
    },
)

gym.register(
    id="Isaac-Pace-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pace.g1_pace_env_cfg:G1PaceEnvCfg"
    },
)

gym.register(
    id="Isaac-Pace-Hoku-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pace.hoku_pace_env_cfg:HokuPaceEnvCfg"
    },
)

gym.register(
    id="Isaac-Pace-Hoku-Flat-Left-Ankle-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pace.hoku_flat_left_ankle_env_cfg:HokuFlatLeftAnklePaceEnvCfg"
    },
)

gym.register(
    id="Isaac-Pace-Neura-4NE1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pace.4ne1_pace_env_cfg:Neura4NE1PaceEnvCfg"
    },
)