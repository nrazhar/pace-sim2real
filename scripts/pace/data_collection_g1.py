# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Script to run an environment with zero action agent for G1 robot."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pace agent for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Pace-G1-v0", help="Name of the task.")
parser.add_argument("--min_frequency", type=float, default=0.1, help="Minimum frequency for the chirp signal in Hz.")
parser.add_argument("--max_frequency", type=float, default=3.0, help="Maximum frequency for the chirp signal in Hz.")
parser.add_argument("--duration", type=float, default=20.0, help="Duration of the chirp signal in seconds.")
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
    # Note: These values are initial guesses/defaults for identification
    # We define them per-joint based on G1 specifications (e.g. G1_29DOF_CFG)
    
    armature_vals = []
    damping_vals = []
    friction_vals = []
    bias_vals = []
    
    for name in joint_order:
        # Defaults (Hips usually)
        # Ref: G1_29DOF_CFG uses Armature=0.03, Damping=2.5 for Hips
        a_val = 0.03
        d_val = 2.5
        f_val = 0.0
        b_val = 0.0
        
        if "knee" in name:
            # Ref: G1_29DOF_CFG uses Damping=5.0 for Knees
            d_val = 5.0
        elif "ankle_pitch" in name:
            # Ref: G1_29DOF_CFG uses Damping=0.2 for Ankle Pitch
            d_val = 0.2
        elif "ankle_roll" in name:
            # Ref: G1_29DOF_CFG uses Damping=0.1 for Ankle Roll
            d_val = 0.1
        elif "waist" in name:
            # Ref: G1_29DOF_CFG uses Damping=5.0, Armature=0.001 for Waist
            d_val = 5.0
            a_val = 0.001
        elif "shoulder" in name or "elbow" in name or "wrist" in name:
            # Ref: G1_29DOF_CFG uses Damping=10.0, Armature=0.001 for Arms
            d_val = 10.0
            a_val = 0.001
        elif "hand" in name:
            # Ref: G1_29DOF_CFG uses Damping=2.0, Armature=0.001 for Hands
            d_val = 2.0
            a_val = 0.001
            
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
    # Note: joint_ids argument ensures we only write to the target joints
    articulation.write_joint_armature_to_sim(armature, joint_ids=joint_ids, env_ids=torch.arange(len(armature)))
    articulation.data.default_joint_armature[:, joint_ids] = armature
    
    articulation.write_joint_viscous_friction_coefficient_to_sim(damping, joint_ids=joint_ids, env_ids=torch.arange(len(damping)))
    articulation.data.default_joint_viscous_friction_coeff[:, joint_ids] = damping
    
    articulation.write_joint_friction_coefficient_to_sim(friction, joint_ids=joint_ids, env_ids=torch.tensor([0]))
    articulation.data.default_joint_friction_coeff[:, joint_ids] = friction

    # Update actuator specific params (delay, bias)
    drive_types = articulation.actuators.keys()
    for drive_type in drive_types:
        # This logic attempts to match the actuator group to our target joints
        drive_indices = articulation.actuators[drive_type].joint_indices
        
        # Handle slice or tensor indices
        if isinstance(drive_indices, slice):
             all_idx = torch.arange(len(all_joint_names), device=joint_ids.device)
             drive_indices_tensor = all_idx[drive_indices]
        else:
             drive_indices_tensor = drive_indices

        # Check intersection between this actuator group and our target joints
        # We only update if there is an overlap
        common_indices = torch.isin(drive_indices_tensor, joint_ids)
        if common_indices.any():
             # Find which of our target params correspond to these actuator joints
             # This requires mapping back from global joint index to our local 'joint_ids' index
             # For simplicity here, we apply the scalar/vector updates if the actuator group is fully within targets
             # or we just update the subset. 
             # Pace Actuator implementation expects matching sizes usually.
             
             # For robust data collection, we assume the standard PaceDCMotorCfg handles update_time_lags
             # effectively broadcasting or matching internally.
             try:
                articulation.actuators[drive_type].update_time_lags(time_lag)
                # Bias update might need careful indexing if bias varies per joint
                # articulation.actuators[drive_type].update_encoder_bias(bias) 
                # Skipping bias update on actuator object for now to avoid shape mismatch if groups differ
                articulation.actuators[drive_type].reset(torch.arange(env.unwrapped.num_envs))
             except Exception as e:
                 print(f"[WARNING]: Could not update actuator params for {drive_type}: {e}")

    # Print configured parameters
    print("\n[INFO] Verifying Joint Parameters (Env 0):")
    print(f"{'Joint Name':<30} | {'Stiffness':<10} | {'Damping':<10} | {'Armature':<10} | {'Friction':<10}")
    print("-" * 85)
    
    # Access properties for target joints
    # Note: We use the data buffers we just wrote to
    stiffness = articulation.data.default_joint_stiffness[0, joint_ids]
    damping_sim = articulation.data.default_joint_viscous_friction_coeff[0, joint_ids]
    armature_sim = articulation.data.default_joint_armature[0, joint_ids]
    friction_sim = articulation.data.default_joint_friction_coeff[0, joint_ids]

    for i, name in enumerate(joint_order):
        s = stiffness[i].item()
        d = damping_sim[i].item()
        a = armature_sim[i].item()
        f = friction_sim[i].item()
        print(f"{name:<30} | {s:<10.4f} | {d:<10.4f} | {a:<10.4f} | {f:<10.4f}")
    print("-" * 85 + "\n")

    data_dir = project_root() / "data" / env_cfg.sim2real.robot_name
    
    # --- Trajectory Generation ---
    duration = args_cli.duration
    sample_rate = 1 / env.unwrapped.sim.get_physics_dt()
    num_steps = int(duration * sample_rate)
    t = torch.linspace(0, duration, steps=num_steps, device=env.unwrapped.device)
    f0 = args_cli.min_frequency
    f1 = args_cli.max_frequency

    # Chirp signal
    phase = 2 * pi * (f0 * t + ((f1 - f0) / (2 * duration)) * t ** 2)
    chirp_signal = torch.sin(phase)

    # Initialize trajectory for target joints
    trajectory = torch.zeros((num_steps, num_target_joints), device=env.unwrapped.device)
    trajectory[:, :] = chirp_signal.unsqueeze(-1)
    
    # Robustly determine params based on joint names
    # We generate directions, bias, and scales dynamically to match the joint_order
    trajectory_directions = []
    trajectory_bias = []
    trajectory_scale = []
    
    print(f"[INFO]: Configuring trajectory for {num_target_joints} joints...")
    
    for name in joint_order:
        # Defaults
        direction = 1.0
        joint_bias = 0.0
        scale = 0.1 # Conservative default
        
        # Determine Side and Direction
        # For base stability, we generally want anti-symmetric motions (e.g. left leg fwd, right leg back)
        # Since G1 limits often share signs (e.g. Knee 0..165), sending -command to Right achieves this.
        is_right = "right" in name
        if is_right:
            direction = -1.0
            
        # Determine Joint Type and Params based on limits
        if "hip_yaw" in name:
            # Limits: ~[-2.7, 2.7]
            scale = 0.2  # Reduced from 0.3
        elif "hip_roll" in name:
            # Limits: L[-0.5, 2.9], R[-2.9, 0.5] (approx)
            # We want to bias outwards (Abduction)
            scale = 0.2
            joint_bias = 0.5 
        elif "hip_pitch" in name:
            # Limits: ~[-2.5, 2.8]
            scale = 0.15  # Reduced from 0.3 (High overshoot observed)
        elif "knee" in name:
            # Limits: [-0.09, 2.88]
            # Must stay positive.
            scale = 0.2  # Reduced from 0.3
            if is_right:
                joint_bias = -1.5
            else:
                joint_bias = 1.5
        elif "ankle_pitch" in name:
            # Limits: ~[-0.8, 0.5]
            scale = 0.15  # Reduced from 0.25
        elif "ankle_roll" in name:
            # Limits: ~[-0.26, 0.26]
            scale = 0.08 # Reduced from 0.1
        elif "waist_yaw" in name:
            # Limits: [-2.6, 2.6]
            scale = 0.3  # Reduced from 0.5
            direction = 1.0 # Center
        elif "waist_roll" in name or "waist_pitch" in name:
            # Limits: ~[-0.5, 0.5]
            scale = 0.1
            direction = 1.0
        elif "shoulder_pitch" in name:
            # Limits: ~[-3.0, 2.6]
            scale = 0.3
        elif "shoulder_roll" in name:
            # Limits: ~[-1.5, 2.2]
            scale = 0.3
        elif "shoulder_yaw" in name:
            # Limits: ~[-2.6, 2.6]
            scale = 0.3
        elif "elbow" in name:
            # Limits: [-1.0, 2.1]
            scale = 0.3
            joint_bias = 0.5
        elif "wrist" in name:
            # Limits: ~[-1.5, 1.5] usually
            scale = 0.2
        elif "hand" in name:
            scale = 0.1

        trajectory_directions.append(direction)
        trajectory_bias.append(joint_bias)
        trajectory_scale.append(scale)
        print(f"  - {name}: Dir={direction}, Bias={joint_bias}, Scale={scale}")

    trajectory_directions = torch.tensor(trajectory_directions, device=env.unwrapped.device)
    trajectory_bias = torch.tensor(trajectory_bias, device=env.unwrapped.device)
    trajectory_scale = torch.tensor(trajectory_scale, device=env.unwrapped.device)
    
    trajectory = (trajectory + trajectory_bias.unsqueeze(0)) * trajectory_directions.unsqueeze(0) * trajectory_scale.unsqueeze(0)

    # --- Initialization ---
    # Write initial position for target joints
    # Crucial: Pass joint_ids to only write these joints
    articulation.write_joint_position_to_sim(
        trajectory[0, :].unsqueeze(0) + bias[0, :], 
        joint_ids=joint_ids
    )
    articulation.write_joint_velocity_to_sim(
        torch.zeros((1, num_target_joints), device=env.unwrapped.device),
        joint_ids=joint_ids
    )

    counter = 0
    dof_pos_buffer = torch.zeros(num_steps, num_target_joints, device=env.unwrapped.device)
    dof_target_pos_buffer = torch.zeros(num_steps, num_target_joints, device=env.unwrapped.device)
    time_data = t
    
    # Prepare a full action tensor
    # If the environment expects actions for all 43 joints, we must provide them.
    # We will use the robot's default joint positions for the non-active joints.
    default_actions = articulation.data.default_joint_pos.clone()
    
    print("[INFO]: Starting Simulation Loop...")
    while simulation_app.is_running():
        with torch.inference_mode():
            # Record current state (only for target joints)
            current_pos = env.unwrapped.scene.articulations["robot"].data.joint_pos[0, joint_ids]
            dof_pos_buffer[counter, :] = current_pos - bias[0]
            
            # Construct Action
            # Start with defaults for all joints
            actions = default_actions.repeat(env.unwrapped.num_envs, 1)
            
            # Update target joints with trajectory
            target_val = trajectory[counter % num_steps, :]
            actions[:, joint_ids] = target_val
            
            # Step environment
            obs, _, _, _, _ = env.step(actions)
            
            # Record target (what we sent)
            # Note: reading back joint_pos_target from sim might be delayed or processed
            # We can log 'target_val' directly or read from sim.
            sim_target = env.unwrapped.scene.articulations["robot"]._data.joint_pos_target[0, joint_ids]
            dof_target_pos_buffer[counter, :] = sim_target
            
            counter += 1
            if counter % 400 == 0:
                print(f"[INFO]: Step {counter} / {num_steps} ({counter/sample_rate:.2f}s)")
            if counter >= num_steps:
                break

    # close the simulator
    env.close()

    from time import sleep
    sleep(1)

    # --- Save Data ---
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = data_dir / timestamp
    (data_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO]: Saving data and plots to {data_dir}")
    
    torch.save({
        "time": time_data.cpu(),
        "dof_pos": dof_pos_buffer.cpu(),
        "des_dof_pos": dof_target_pos_buffer.cpu(),
        "joint_names": joint_order
    }, data_dir / "chirp_data.pt")

    import matplotlib.pyplot as plt
    
    # Save configuration to text file
    config_path = data_dir / "config.txt"
    with open(config_path, "w") as f:
        f.write(f"Data Collection Config - {timestamp}\n")
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
        f.write(f"{'Joint Name':<30} | {'Stiffness':<10} | {'Damping':<10} | {'Armature':<10} | {'Friction':<10}\n")
        f.write("-" * 85 + "\n")
        
        stiffness = articulation.data.default_joint_stiffness[0, joint_ids]
        damping_sim = articulation.data.default_joint_viscous_friction_coeff[0, joint_ids]
        armature_sim = articulation.data.default_joint_armature[0, joint_ids]
        friction_sim = articulation.data.default_joint_friction_coeff[0, joint_ids]

        for i, name in enumerate(joint_order):
            s = stiffness[i].item()
            d = damping_sim[i].item()
            a = armature_sim[i].item()
            fr = friction_sim[i].item()
            f.write(f"{name:<30} | {s:<10.4f} | {d:<10.4f} | {a:<10.4f} | {fr:<10.4f}\n")
        
        f.write("\nTrajectory Configuration:\n")
        f.write(f"{'Joint Name':<30} | {'Direction':<10} | {'Bias':<10} | {'Scale':<10}\n")
        f.write("-" * 70 + "\n")
        for i, name in enumerate(joint_order):
             f.write(f"{name:<30} | {trajectory_directions[i].item():<10.1f} | {trajectory_bias[i].item():<10.2f} | {trajectory_scale[i].item():<10.2f}\n")
    
        # Add Safety Analysis to config file
        f.write("\nSafety Analysis of Collected Data:\n")
        f.write(f"{'Joint Name':<30} | {'Min Pos':<10} | {'Max Pos':<10} | {'Min Target':<10} | {'Max Target':<10} | {'Max Error':<10}\n")
        f.write("-" * 90 + "\n")

        for i, name in enumerate(joint_order):
            measured = dof_pos_buffer[:, i]
            target = dof_target_pos_buffer[:, i]
            error = torch.abs(measured - target)
            
            f.write(f"{name:<30} | {measured.min():<10.4f} | {measured.max():<10.4f} | {target.min():<10.4f} | {target.max():<10.4f} | {error.max():<10.4f}\n")
    
    print(f"[INFO]: Saved configuration to {config_path}")

    # Get joint limits from the robot
    dof_limits_lower = articulation.data.soft_joint_pos_limits[0, :, 0]
    dof_limits_upper = articulation.data.soft_joint_pos_limits[0, :, 1]

    for i in range(num_target_joints):
        plt.figure()
        joint_name = joint_order[i]
        
        # Find index in full articulation to get limits
        if joint_name in all_joint_names:
            idx = all_joint_names.index(joint_name)
            lower_limit = dof_limits_lower[idx].item()
            upper_limit = dof_limits_upper[idx].item()
        else:
            lower_limit, upper_limit = -float('inf'), float('inf')

        plt.plot(t.cpu().numpy(), dof_pos_buffer[:, i].cpu().numpy(), label=f"{joint_name} pos")
        plt.plot(t.cpu().numpy(), dof_target_pos_buffer[:, i].cpu().numpy(), label=f"{joint_name} target", linestyle='dashed')
        
        # Add safety limits to plot
        plt.axhline(y=lower_limit, color='r', linestyle=':', label='Lower Limit')
        plt.axhline(y=upper_limit, color='r', linestyle=':', label='Upper Limit')
        
        plt.title(f"Joint {joint_name} Trajectory")
        plt.xlabel("Time [s]")
        plt.ylabel("Joint position [rad]")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plot_path = data_dir / f"chirp_plot_{timestamp}_{joint_name}.png"
        plt.savefig(plot_path)
        plt.close()

    # Print Safety Analysis
    print("\n[INFO] Safety Analysis of Collected Data:")
    print(f"{'Joint Name':<30} | {'Min Pos':<10} | {'Max Pos':<10} | {'Min Target':<10} | {'Max Target':<10} | {'Max Error':<10}")
    print("-" * 90)

    for i, name in enumerate(joint_order):
        measured = dof_pos_buffer[:, i]
        target = dof_target_pos_buffer[:, i]
        error = torch.abs(measured - target)
        
        print(f"{name:<30} | {measured.min():.4f}     | {measured.max():.4f}     | {target.min():.4f}       | {target.max():.4f}       | {error.max():.4f}")

if __name__ == "__main__":
    main()
    simulation_app.close()

