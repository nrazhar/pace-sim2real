# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pace agent for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Pace-Anymal-D-v0", help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from torch import pi

import pace_sim2real.tasks  # noqa: F401
from pace_sim2real.utils.paths import project_root
from pace_sim2real.optim import CMAESOptimizer

IDENTIFIED_JOINTS = [
    "LF_HAA",
    "LF_HFE",
    "LF_KFE",
    "RF_HAA",
    "RF_HFE",
    "RF_KFE",
    "LH_HAA",
    "LH_HFE",
    "LH_KFE",
    "RH_HAA",
    "RH_HFE",
    "RH_KFE",
]


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    opt = CMAESOptimizer(
        initial_params=None,
        sigma=0.5,
        population_size=env.unwrapped.num_envs
    )  # TODO: implement optimization loop

    articulation = env.unwrapped.scene["robot"]
    joint_names = IDENTIFIED_JOINTS
    joint_ids = [articulation.joint_names.index(name) for name in joint_names]

    project_data_dir = "anymal_sim"
    data_dir = project_root() / "data" / project_data_dir

    data = torch.load(data_dir / "chirp_data.pt")
    time_data = data["time"].to(env.unwrapped.device)
    target_dof_pos = data["dof_target_pos"].to(env.unwrapped.device)
    measured_dof_pos = data["dof_pos"].to(env.unwrapped.device)
    num_steps = time_data.shape[0]
    sim_dt = env.unwrapped.sim.cfg.dt

    # reset environment with first position from data
    tmp = env.unwrapped.scene.articulations["robot"]._data.default_joint_pos
    env.unwrapped.scene.articulations["robot"]._data.default_joint_pos[0, joint_ids] = measured_dof_pos[0, :]
    print(measured_dof_pos[0, :])
    env.reset()
    env.unwrapped.scene.articulations["robot"]._data.default_joint_pos = tmp

    counter = 0
    # simulate environment
    dof_pos_buffer = torch.zeros(num_steps, len(joint_ids), device=env.unwrapped.device)
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            actions[:, joint_ids] = target_dof_pos[counter, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
            # apply actions
            env.step(actions)
            dof_pos_buffer[counter, :] = env.unwrapped.scene.articulations["robot"].data.joint_pos[0, joint_ids]
            counter += 1
            if counter % 100 == 0:
                print(f"[INFO]: Step {counter * sim_dt:.1f} / {time_data[-1]:.1f} seconds ({counter / num_steps * 100:.1f} %)")
            if counter >= num_steps:
                print("[INFO]: Reached the end of the trajectory, exiting.")
                break

    # close the simulator
    env.close()

    import matplotlib.pyplot as plt

    for i in range(len(joint_ids)):
        plt.figure()
        plt.plot(dof_pos_buffer[:, i].cpu().numpy(), label=f"{joint_names[i]} Sim")
        plt.plot(target_dof_pos[:, i].cpu().numpy(), label=f"{joint_names[i]} Target", linestyle='dashed')
        plt.plot(measured_dof_pos[:, i].cpu().numpy(), label=f"{joint_names[i]} Measured", linestyle='dotted')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
