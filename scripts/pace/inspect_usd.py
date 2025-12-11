"""Script to inspect G1 USD file for joint limits and other info.

Usage:
    python scripts/inspect_g1_usd.py
"""

import argparse
import os
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Inspect USD.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now we can import pxr and other isaac libraries
from pxr import Usd, UsdPhysics, UsdGeom
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


def main():
    # Path to G1 USD as defined in unitree.py
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd"
    # usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-D/anymal_d.usd"

    # usd_path = "/home/azhar/ws/n_robotics_robot_description/mujoco_robot/mujoco_robot.usd"

    print(f"[INFO] Opening USD stage: {usd_path}")
    
    # Open the stage
    try:
        stage = Usd.Stage.Open(usd_path)
    except Exception as e:
        print(f"[ERROR] Failed to open stage: {e}")
        return

    if not stage:
        print(f"[ERROR] Failed to open stage: {usd_path}")
        return

    # Collect data first
    joints_data = []
    joint_names = []
    body_names = set()

    # Iterate over all prims
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            name = prim.GetName()
            type_name = prim.GetTypeName()
            
            joint_names.append(name)

            # Get Body targets
            joint = UsdPhysics.Joint(prim)
            body0_targets = joint.GetBody0Rel().GetTargets()
            body0 = body0_targets[0].name if body0_targets else "N/A"
            if body0 != "N/A": body_names.add(body0)

            body1_targets = joint.GetBody1Rel().GetTargets()
            body1 = body1_targets[0].name if body1_targets else "N/A"
            if body1 != "N/A": body_names.add(body1)
            
            # Default values
            lower = "N/A"
            upper = "N/A"
            max_force = "N/A"
            max_vel = "N/A"

            # Try to get Physics limits
            # Revolute Joints often have limits in 'limit:rotX:physics:low' etc depending on axis
            # But we can search for attributes
            
            attrs = prim.GetAttributes()
            for attr in attrs:
                attr_name = attr.GetName()
                val = attr.Get()
                
                # Check for limits (case insensitive)
                attr_lower = attr_name.lower()
                
                if "limit" in attr_lower:
                    if "low" in attr_lower:
                        lower = f"{val:.2f}"
                    elif "high" in attr_lower or "upper" in attr_lower:
                        upper = f"{val:.2f}"
                
                # Drive API properties (often on the joint or a child)
                # In Isaac Sim, DriveAPI is usually applied to the joint
                if "drive" in attr_lower:
                    if "maxforce" in attr_lower:
                        max_force = f"{val:.2f}"
                
                # Max Velocity might be in physxJoint:maxJointVelocity or similar
                if "maxjointvelocity" in attr_lower or ("maxvelocity" in attr_lower and "drive" not in attr_lower):
                     max_vel = f"{val:.2f}"
            
            joints_data.append({
                "name": name,
                "type": type_name,
                "body0": body0,
                "body1": body1,
                "lower": lower,
                "upper": upper,
                "max_force": max_force,
                "max_vel": max_vel
            })

    # Print summary lists
    print("\nJoint List:")
    print(joint_names)

    print("\nBody List:")
    print(sorted(list(body_names)))
    print("\n")

    print("=" * 160)
    print(f"{'Joint Name':<30} | {'Type':<15} | {'Body 0':<20} | {'Body 1':<20} | {'Lower':<10} | {'Upper':<10} | {'Max Force':<10} | {'Max Vel':<10}")
    print("=" * 160)
    
    for j in joints_data:
        print(f"{j['name']:<30} | {j['type']:<15} | {j['body0']:<20} | {j['body1']:<20} | {j['lower']:<10} | {j['upper']:<10} | {j['max_force']:<10} | {j['max_vel']:<10}")

    print("=" * 160)

    # Keep the app open if not headless so user can see it
    if not args_cli.headless:
        print("\n[INFO] Simulation is running. Press Enter to close...")
        input()

if __name__ == "__main__":
    main()
    simulation_app.close()

