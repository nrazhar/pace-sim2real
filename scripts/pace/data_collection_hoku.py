# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic (Adapted for Hoku)
# Licensed under the Apache License 2.0

"""Script to run an environment with zero action agent for Hoku robot."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pace agent for Isaac Lab environments.")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument(
    "--task", type=str, default="Isaac-Pace-Hoku-Flat-Left-Ankle-v0", help="Name of the task."
)
parser.add_argument(
    "--min_frequency",
    type=float,
    default=0.0,
    help="Minimum frequency for the chirp signal in Hz.",
)
parser.add_argument(
    "--max_frequency",
    type=float,
    default=3.0,
    help="Maximum frequency for the chirp signal in Hz.",
)
parser.add_argument(
    "--duration",
    type=float,
    default=20.0,
    help="Duration of the chirp signal in seconds.",
)
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
from pace_sim2real.utils import project_root


def main():
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()

    articulation = env.unwrapped.scene["robot"]

    # Get target joints from configuration
    joint_order = env_cfg.sim2real.joint_order
    print(f"[INFO]: Target joints for data collection: {joint_order}")

    # Map joint names to indices
    all_joint_names = articulation.joint_names
    joint_ids = []
    for name in joint_order:
        if name in all_joint_names:
            joint_ids.append(all_joint_names.index(name))
        else:
            print(f"[WARNING]: Joint {name} not found in robot articulation!")

    joint_ids = torch.tensor(joint_ids, device=env.unwrapped.device)
    num_target_joints = len(joint_ids)

    # Set up Pace-specific motor parameters (Armature, Damping, etc.)
    armature_vals = []
    damping_vals = []
    friction_vals = []
    bias_vals = []

    for name in joint_order:
        # Defaults
        a_val = 0.03
        d_val = 4.5
        f_val = 0.0
        b_val = 0.0

        # Logic based on Hoku naming convention (x, y, z, inner, outer)
        if "knee" in name:
            d_val = 4.5
        elif "ankle" in name:
             # ankle_inner, ankle_outer, ankle_x, etc.
            d_val = 4.0
        elif "torso" in name:
            # Torso usually has lower armature/damping if it's a spinal joint
            a_val = 0.01
            d_val = 5.0
        elif "shoulder" in name or "elbow" in name or "wrist" in name:
            a_val = 0.005 # Lower armature for arms
            d_val = 2.0   # Lower damping for arms
        
        armature_vals.append(a_val)
        damping_vals.append(d_val)
        friction_vals.append(f_val)
        bias_vals.append(b_val)

    armature = torch.tensor(armature_vals, device=env.unwrapped.device).unsqueeze(0)
    damping = torch.tensor(damping_vals, device=env.unwrapped.device).unsqueeze(0)
    friction = torch.tensor(friction_vals, device=env.unwrapped.device).unsqueeze(0)
    bias = torch.tensor(bias_vals, device=env.unwrapped.device).unsqueeze(0)
    time_lag = torch.tensor([[5]], dtype=torch.int, device=env.unwrapped.device)

    # Write parameters to sim
    articulation.write_joint_armature_to_sim(
        armature, joint_ids=joint_ids, env_ids=torch.arange(len(armature))
    )
    articulation.data.default_joint_armature[:, joint_ids] = armature

    articulation.write_joint_viscous_friction_coefficient_to_sim(
        damping, joint_ids=joint_ids, env_ids=torch.arange(len(damping))
    )
    articulation.data.default_joint_viscous_friction_coeff[:, joint_ids] = damping

    articulation.write_joint_friction_coefficient_to_sim(
        friction, joint_ids=joint_ids, env_ids=torch.tensor([0])
    )
    articulation.data.default_joint_friction_coeff[:, joint_ids] = friction

    # Update actuator specific params (delay, bias)
    drive_types = articulation.actuators.keys()
    for drive_type in drive_types:
        drive_indices = articulation.actuators[drive_type].joint_indices
        if isinstance(drive_indices, slice):
            all_idx = torch.arange(len(all_joint_names), device=joint_ids.device)
            drive_indices_tensor = all_idx[drive_indices]
        else:
            drive_indices_tensor = drive_indices

        common_indices = torch.isin(drive_indices_tensor, joint_ids)
        if common_indices.any():
            try:
                articulation.actuators[drive_type].update_time_lags(time_lag)
                articulation.actuators[drive_type].reset(
                    torch.arange(env.unwrapped.num_envs)
                )
            except Exception as e:
                print(
                    f"[WARNING]: Could not update actuator params for {drive_type}: {e}"
                )

    # Print configured parameters
    print("\n[INFO] Verifying Joint Parameters (Env 0):")
    header_fmt = "{:<30} | {:<10} | {:<10} | {:<10} | {:<12} | {:<10} | {:<10} | {:<5}"
    print(header_fmt.format("Joint Name", "Stiffness", "Damping", "Armature", "Visc. Damp", "Friction", "Bias", "Lag"))
    print("-" * 120)

    stiffness_sim = articulation.data.default_joint_stiffness[0, joint_ids]
    damping_gain_sim = articulation.data.default_joint_damping[0, joint_ids]
    viscous_sim = articulation.data.default_joint_viscous_friction_coeff[0, joint_ids]
    armature_sim = articulation.data.default_joint_armature[0, joint_ids]
    friction_sim = articulation.data.default_joint_friction_coeff[0, joint_ids]

    for i, name in enumerate(joint_order):
        s = stiffness_sim[i].item()
        d = damping_gain_sim[i].item()
        a = armature_sim[i].item()
        v = viscous_sim[i].item()
        f = friction_sim[i].item()
        b = bias_vals[i]
        l = time_lag.item()

        print(header_fmt.format(name, f"{s:.4f}", f"{d:.4f}", f"{a:.4f}", f"{v:.4f}", f"{f:.4f}", f"{b:.4f}", f"{l}"))
        print("-" * 120 + "\n")

    data_dir = project_root() / "data" / env_cfg.sim2real.robot_name

    # --- Trajectory Generation ---
    duration = args_cli.duration
    sample_rate = 1 / env.unwrapped.sim.get_physics_dt()
    num_steps = int(duration * sample_rate)
    t = torch.linspace(0, duration, steps=num_steps, device=env.unwrapped.device)
    f0 = args_cli.min_frequency
    f1 = args_cli.max_frequency

    # Chirp signal
    phase = 2 * pi * (f0 * t + ((f1 - f0) / (2 * duration)) * t**2)
    chirp_signal = torch.sin(phase)

    # Initialize trajectory
    trajectory = torch.zeros(
        (num_steps, num_target_joints), device=env.unwrapped.device
    )
    trajectory[:, :] = chirp_signal.unsqueeze(-1)

    trajectory_directions = []
    trajectory_bias = []
    trajectory_scale = []

    print(f"[INFO]: Configuring trajectory for {num_target_joints} joints (Hoku naming)...")

    for name in joint_order:
        # Defaults
        direction = 1.0
        joint_bias = 0.0
        scale = 0.1

        # Determine Side
        is_right = "right" in name
        if is_right:
            direction = -1.0
        # Assign parameters based on joint type
        if "hip_yaw" in name:
            # Target: [-0.25, 0.25]
            joint_bias = 0.0
            scale = 0.75
        elif "hip_roll" in name:
            # Target: [-0.1, 0.1]
            if is_right:
                joint_bias = -0.2
            else:
                joint_bias = 0.2
            scale = 0.15
        elif "hip_pitch" in name:
            # Target: [-0.2, 0.0]
            joint_bias = 0.0
            scale = 0.75
        elif "knee_pitch" in name:
            # Target: [0.2, 0.8] -> Center is 0.5, Range is +/- 0.3
            joint_bias = 0.5
            scale = 0.3
        elif "ankle_pitch" in name:
            # Target: [-0.3, -0.1] -> Center is -0.2
            joint_bias = 0.0
            scale = 0.25
        elif "ankle_roll" in name:
            # Target: [-0.1, 0.1]
            joint_bias = 0.0
            scale = 0.25

        trajectory_directions.append(direction)
        trajectory_bias.append(joint_bias)
        trajectory_scale.append(scale)
        print(f"  - {name}: Dir={direction}, Bias={joint_bias}, Scale={scale}")

    trajectory_directions = torch.tensor(
        trajectory_directions, device=env.unwrapped.device
    )
    trajectory_bias = torch.tensor(trajectory_bias, device=env.unwrapped.device)
    trajectory_scale = torch.tensor(trajectory_scale, device=env.unwrapped.device)

    trajectory = trajectory * trajectory_directions.unsqueeze(
        0
    ) * trajectory_scale.unsqueeze(0) + trajectory_bias.unsqueeze(0)

    # --- Initialization ---
    articulation.write_joint_position_to_sim(
        trajectory[0, :].unsqueeze(0) + bias[0, :], joint_ids=joint_ids
    )
    articulation.write_joint_velocity_to_sim(
        torch.zeros((1, num_target_joints), device=env.unwrapped.device),
        joint_ids=joint_ids,
    )

    counter = 0
    dof_pos_buffer = torch.zeros(
        num_steps, num_target_joints, device=env.unwrapped.device
    )
    dof_target_pos_buffer = torch.zeros(
        num_steps, num_target_joints, device=env.unwrapped.device
    )
    time_data = t

    default_actions = articulation.data.default_joint_pos.clone()

    print("[INFO]: Starting Simulation Loop...")
    while simulation_app.is_running():
        with torch.inference_mode():
            current_pos = env.unwrapped.scene.articulations["robot"].data.joint_pos[
                0, joint_ids
            ]
            dof_pos_buffer[counter, :] = current_pos - bias[0]

            actions = default_actions.repeat(env.unwrapped.num_envs, 1)
            target_val = trajectory[counter % num_steps, :]
            actions[:, joint_ids] = target_val

            obs, _, _, _, _ = env.step(actions)

            sim_target = env.unwrapped.scene.articulations[
                "robot"
            ]._data.joint_pos_target[0, joint_ids]
            dof_target_pos_buffer[counter, :] = sim_target

            counter += 1
            if counter % 400 == 0:
                print(
                    f"[INFO]: Step {counter} / {num_steps} ({counter/sample_rate:.2f}s)"
                )
            if counter >= num_steps:
                break

    env.close()

    from time import sleep
    sleep(1)

    # --- Save Data ---
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = data_dir / timestamp
    (data_dir).mkdir(parents=True, exist_ok=True)

    print(f"[INFO]: Saving data and plots to {data_dir}")

    torch.save(
        {
            "time": time_data.cpu(),
            "dof_pos": dof_pos_buffer.cpu(),
            "des_dof_pos": dof_target_pos_buffer.cpu(),
            "joint_names": joint_order,
        },
        data_dir / "chirp_data.pt",
    )

    import matplotlib.pyplot as plt

    config_path = data_dir / "config.txt"
    with open(config_path, "w") as f:
        f.write(f"Data Collection Config (Hoku) - {timestamp}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Simulation Settings:\n")
        f.write(f"  Duration: {args_cli.duration} s\n")
        f.write(f"  Min Frequency: {args_cli.min_frequency} Hz\n")
        f.write(f"  Max Frequency: {args_cli.max_frequency} Hz\n")
        f.write(f"  Sample Rate: {sample_rate:.2f} Hz\n")
        f.write(f"  Task Name: {args_cli.task}\n")
        f.write(f"  Robot Name: {env_cfg.sim2real.robot_name}\n\n")

        f.write("Target Joints for Testing:\n")
        for i, name in enumerate(joint_order):
            f.write(f"  - {name}\n")
        f.write("\n")

        f.write("Joint Parameters:\n")
        f.write(
            f"{'Joint Name':<30} | {'Stiffness':<10} | {'Damping':<10} | {'Armature':<10} | {'Friction':<10}\n"
        )
        f.write("-" * 85 + "\n")

        stiffness = articulation.data.default_joint_stiffness[0, joint_ids]
        damping_sim = articulation.data.default_joint_viscous_friction_coeff[
            0, joint_ids
        ]
        armature_sim = articulation.data.default_joint_armature[0, joint_ids]
        friction_sim = articulation.data.default_joint_friction_coeff[0, joint_ids]

        for i, name in enumerate(joint_order):
            s = stiffness[i].item()
            d = damping_sim[i].item()
            a = armature_sim[i].item()
            fr = friction_sim[i].item()
            f.write(
                f"{name:<30} | {s:<10.4f} | {d:<10.4f} | {a:<10.4f} | {fr:<10.4f}\n"
            )

        f.write("\nTrajectory Configuration:\n")
        f.write(
            f"{'Joint Name':<30} | {'Direction':<10} | {'Bias':<10} | {'Scale':<10}\n"
        )
        f.write("-" * 70 + "\n")
        for i, name in enumerate(joint_order):
            f.write(
                f"{name:<30} | {trajectory_directions[i].item():<10.1f} | {trajectory_bias[i].item():<10.2f} | {trajectory_scale[i].item():<10.2f}\n"
            )

        f.write("\nSafety Analysis of Collected Data:\n")
        f.write(
            f"{'Joint Name':<30} | {'Min Pos':<10} | {'Max Pos':<10} | {'Min Target':<10} | {'Max Target':<10} | {'Max Error':<10}\n"
        )
        f.write("-" * 90 + "\n")

        for i, name in enumerate(joint_order):
            measured = dof_pos_buffer[:, i]
            target = dof_target_pos_buffer[:, i]
            error = torch.abs(measured - target)

            f.write(
                f"{name:<30} | {measured.min():<10.4f} | {measured.max():<10.4f} | {target.min():<10.4f} | {target.max():<10.4f} | {error.max():<10.4f}\n"
            )

    print(f"[INFO]: Saved configuration to {config_path}")

    dof_limits_lower = articulation.data.soft_joint_pos_limits[0, :, 0]
    dof_limits_upper = articulation.data.soft_joint_pos_limits[0, :, 1]

    for i in range(num_target_joints):
        plt.figure()
        joint_name = joint_order[i]

        if joint_name in all_joint_names:
            idx = all_joint_names.index(joint_name)
            lower_limit = dof_limits_lower[idx].item()
            upper_limit = dof_limits_upper[idx].item()
        else:
            lower_limit, upper_limit = -float("inf"), float("inf")

        plt.plot(
            t.cpu().numpy(),
            dof_pos_buffer[:, i].cpu().numpy(),
            label=f"{joint_name} pos",
        )
        plt.plot(
            t.cpu().numpy(),
            dof_target_pos_buffer[:, i].cpu().numpy(),
            label=f"{joint_name} target",
            linestyle="dashed",
        )

        plt.axhline(y=lower_limit, color="r", linestyle=":", label="Lower Limit")
        plt.axhline(y=upper_limit, color="r", linestyle=":", label="Upper Limit")

        plt.title(f"Joint {joint_name} Trajectory")
        plt.xlabel("Time [s]")
        plt.ylabel("Joint position [rad]")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plot_path = data_dir / f"chirp_plot_{timestamp}_{joint_name}.png"
        plt.savefig(plot_path)
        plt.close()

    print("\n[INFO] Safety Analysis of Collected Data:")
    print(
        f"{'Joint Name':<30} | {'Lower Lim':<10} | {'Upper Lim':<10} | {'Min Pos':<10} | {'Max Pos':<10} | {'Min Target':<10} | {'Max Target':<10} | {'Max Error':<10}"
    )
    print("-" * 140)

    for i, name in enumerate(joint_order):
        measured = dof_pos_buffer[:, i]
        target = dof_target_pos_buffer[:, i]
        error = torch.abs(measured - target)

        if name in all_joint_names:
            idx = all_joint_names.index(name)
            lower_limit = dof_limits_lower[idx].item()
            upper_limit = dof_limits_upper[idx].item()
        else:
            lower_limit, upper_limit = -999.0, 999.0

        print(
            f"{name:<30} | {lower_limit:<10.4f} | {upper_limit:<10.4f} | {measured.min():<10.4f} | {measured.max():<10.4f} | {target.min():<10.4f} | {target.max():<10.4f} | {error.max():<10.4f}"
        )


if __name__ == "__main__":
    main()
    simulation_app.close()