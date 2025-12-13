# © 2025 ETH Zurich, Robotic Systems Lab
# Adapted for Neura 4NE1
# Licensed under the Apache License 2.0

"""Script to run an environment with zero action agent for Neura 4NE1 robot."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher
import os
from datetime import datetime
# add argparse arguments
parser = argparse.ArgumentParser(description="Pace agent for Isaac Lab environments.")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument(
    "--task", type=str, default="Isaac-Pace-Neura-4NE1-v0", help="Name of the task."
)
parser.add_argument(
    "--min_frequency",
    type=float,
    default=0.05,
    help="Minimum frequency for the chirp signal in Hz.",
)
parser.add_argument(
    "--max_frequency",
    type=float,
    default=5.0,
    help="Maximum frequency for the chirp signal in Hz.",
)
parser.add_argument(
    "--duration",
    type=float,
    default=60.0,
    help="Duration of the chirp signal in seconds.",
)
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length", type=int, default=400, help="Length of the recorded video (in steps)."
)
parser.add_argument(
    "--video_interval", type=int, default=2000, help="Interval between video recordings (in steps)."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    robot_name = env_cfg.sim2real.robot_name
    
    # Path: data/<robot_name>/<timestamp>
    data_dir = project_root() / "data" / robot_name / timestamp
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output Directory: {data_dir}")
    # create environment
    env = gym.make(
        args_cli.task, 
        cfg=env_cfg, 
        render_mode="rgb_array" if args_cli.video else None
    )

    physics_dt = env.unwrapped.sim.get_physics_dt()
    sample_rate = 1.0 / physics_dt  # e.g., 400.0 Hz
    
    # 2. Calculate Total Steps
    total_steps = int(args_cli.duration * sample_rate)
    
    if args_cli.video:
        print(f"[INFO] Configuring Video: {sample_rate} Hz, {total_steps} frames (Full Run)")
        env.metadata["render_fps"] = sample_rate
        
        video_kwargs = {
            "video_folder": str(data_dir),
            # Trigger ONLY at step 0 to get one single file
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            # Record exactly the number of steps in the full duration
            "video_length": args_cli.video_length,
            # Force video FPS to match Sim FPS so 1 sec sim = 1 sec video
            "disable_logger": True,
            "name_prefix": "video"
        }
        
        # Apply wrapper
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()

    articulation = env.unwrapped.scene["robot"]

    # Get target joints from configuration
    # Ensure your G1PaceCfg or equivalent 4NE1 config has this list populated
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
        # Defaults based on 4NE1 size/specs
        a_val = 0.03
        d_val = 5.0
        f_val = 0.0
        b_val = 0.0

        # Logic based on 4NE1 naming convention (x, y, z)
        # 4NE1 uses suffixes like _hip_x, _hip_y rather than _roll, _yaw

        if "knee" in name:
            # Knee requires higher damping/stiffness
            d_val = 14.5
            a_val = 0.03
        elif "hip" in name:
            # Hip joints are heavy
            d_val = 15.0
            a_val = 0.03
        elif "ankle" in name:
            # Ankles are smaller
            d_val = 0.5
            a_val = 0.01
        elif "torso" in name:
            # Torso carries upper body mass
            a_val = 0.02
            d_val = 6.5
        elif "shoulder" in name or "elbow" in name:
            a_val = 0.005
            d_val = 1.0
        elif "wrist" in name:
            a_val = 0.001
            d_val = 0.1

        armature_vals.append(a_val)
        damping_vals.append(d_val)
        friction_vals.append(f_val)
        bias_vals.append(b_val)

    armature = torch.tensor(armature_vals, device=env.unwrapped.device).unsqueeze(0)
    damping = torch.tensor(damping_vals, device=env.unwrapped.device).unsqueeze(0)
    friction = torch.tensor(friction_vals, device=env.unwrapped.device).unsqueeze(0)
    bias = torch.tensor(bias_vals, device=env.unwrapped.device).unsqueeze(0)
    # Default lag 5 steps (~12ms at 400Hz)
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
    print(
        header_fmt.format(
            "Joint Name",
            "Stiffness",
            "Damping",
            "Armature",
            "Visc. Damp",
            "Friction",
            "Bias",
            "Lag",
        )
    )
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

        print(
            header_fmt.format(
                name,
                f"{s:.4f}",
                f"{d:.4f}",
                f"{a:.4f}",
                f"{v:.4f}",
                f"{f:.4f}",
                f"{b:.4f}",
                f"{l}",
            )
        )
        print("-" * 120 + "\n")


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

    print(
        f"[INFO]: Configuring trajectory for {num_target_joints} joints (4NE1 naming)..."
    )

    for name in joint_order:
        # Defaults
        direction = 1.0
        joint_bias = 0.0
        scale = 0.1

        # Determine Side (usually right side roll/yaw is mirrored)
        is_right = "right" in name
        if is_right:
            direction = -1.0

        # 4NE1 Trajectory Logic based on provided limits
        # hip_y (Pitch): -90 to 60.
        # hip_x (Roll/Abd): -8.6 to 33.6 (Left), -33.6 to 8.6 (Right)
        # hip_z (Yaw): -90 to 90
        # knee: -50 to 80
        # ankle_y (Pitch): -35 to 30

        if "hip_z" in name:  # Yaw
            joint_bias = 0.0
            scale = 1.0  # +/- 0.3 rad (~17 deg)

        elif "hip_x" in name:  # Roll / Abduction
            # Center around slight abduction to avoid collision
            joint_bias = 0.25 if not is_right else -0.25
            scale = 0.2  # Keep small

        elif "hip_y" in name:  # Pitch
            # -90 to 60. Safe center -0.2
            joint_bias = -0.2
            scale = 0.75

        elif "knee" in name:  # Knee
            # -50 to 80.
            # If 0 is straight leg, bias to 0.3 (bent) for chirp
            joint_bias = 0.1
            scale = 0.75

        elif "ankle_y" in name:  # Pitch
            # -35 to 30.
            joint_bias = 0.0
            scale = 0.3

        elif "ankle_x" in name:  # Roll
            # -25 to 25.
            joint_bias = 0.0
            scale = 0.3

        elif "torso_z" in name:  # Yaw
            joint_bias = 0.0
            scale = 0.3
        elif "torso_y" in name:  # Pitch (-5 to 20)
            joint_bias = 0.1  # Slight forward lean
            scale = 0.1
        elif "torso_x" in name:  # Roll
            joint_bias = 0.0
            scale = 0.1

        trajectory_directions.append(direction)
        trajectory_bias.append(joint_bias)
        trajectory_scale.append(scale)
        print(f"  - {name}: Dir={direction}, Bias={joint_bias}, Scale={scale}")

    trajectory_directions = torch.tensor(
        trajectory_directions, device=env.unwrapped.device
    )
    trajectory_bias = torch.tensor(trajectory_bias, device=env.unwrapped.device)
    trajectory_scale = torch.tensor(trajectory_scale, device=env.unwrapped.device)

    # Apply direction, scale and bias to chirp
    trajectory = trajectory * trajectory_directions.unsqueeze(
        0
    ) * trajectory_scale.unsqueeze(0) + trajectory_bias.unsqueeze(0)

    # --- Initialization ---
    # Set initial pose to the start of the trajectory + bias
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
        f.write(f"Data Collection Config (Neura 4NE1) - {timestamp}\n")
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
