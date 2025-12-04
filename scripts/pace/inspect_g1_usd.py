"""Script to inspect G1 USD file for joint limits and other info.

Usage:
    python scripts/inspect_g1_usd.py
"""

import argparse
import os
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Inspect G1 USD.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now we can import pxr and other isaac libraries
from pxr import Usd, UsdPhysics, UsdGeom
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

def main():
    # Path to G1 USD as defined in unitree.py
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd"
    
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

    print("=" * 120)
    print(f"{'Joint Name':<30} | {'Type':<15} | {'Lower':<10} | {'Upper':<10} | {'Max Force':<10} | {'Max Vel':<10} | {'Damping':<10} | {'Stiffness':<10}")
    print("=" * 120)

    # Iterate over all prims
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            name = prim.GetName()
            type_name = prim.GetTypeName()
            
            # Default values
            lower = "N/A"
            upper = "N/A"
            max_force = "N/A"
            max_vel = "N/A"
            damping = "N/A"
            stiffness = "N/A"

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
                    elif "damping" in attr_lower:
                        damping = f"{val:.2f}"
                    elif "stiffness" in attr_lower:
                        stiffness = f"{val:.2f}"
                
                # Max Velocity might be in physxJoint:maxJointVelocity or similar
                if "maxjointvelocity" in attr_lower or ("maxvelocity" in attr_lower and "drive" not in attr_lower):
                     max_vel = f"{val:.2f}"
            
            # Check for explicit max velocity in limit API if available?
            # Often it's not explicitly set in USD physics for simple joints unless using PhysX limits
            
            print(f"{name:<30} | {type_name:<15} | {lower:<10} | {upper:<10} | {max_force:<10} | {max_vel:<10} | {damping:<10} | {stiffness:<10}")

    print("=" * 120)

    # Also list all joints found in a simpler list for copy-pasting
    print("\nJoint List:")
    joints = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joints.append(prim.GetName())
    print(joints)

    # Keep the app open if not headless so user can see it
    if not args_cli.headless:
        print("\n[INFO] Simulation is running. Press Enter to close...")
        input()

if __name__ == "__main__":
    main()
    simulation_app.close()

