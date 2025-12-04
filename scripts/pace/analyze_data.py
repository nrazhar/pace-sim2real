
import torch
import os
from pathlib import Path

def project_root():
    return Path(__file__).resolve().parents[2]

def analyze_latest_data():
    data_root = project_root() / "data" / "g1_sim"
    # Find latest directory
    if not data_root.exists():
         print(f"Data root {data_root} does not exist.")
         return

    subdirs = [d for d in data_root.iterdir() if d.is_dir()]
    if not subdirs:
        print("No data directories found.")
        return
    latest_dir = max(subdirs, key=os.path.getmtime)
    print(f"Analyzing data from: {latest_dir}")
    
    data_path = latest_dir / "chirp_data.pt"
    if not data_path.exists():
        print(f"No chirp_data.pt found in {latest_dir}")
        return

    try:
        data = torch.load(data_path, map_location='cpu')
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # Keys: 'time', 'dof_pos', 'des_dof_pos', 'joint_names'
    
    joint_names = data['joint_names']
    dof_pos = data['dof_pos'] # [steps, joints]
    des_dof_pos = data['des_dof_pos'] # [steps, joints]
    
    print(f"{'Joint Name':<30} | {'Min Pos':<10} | {'Max Pos':<10} | {'Min Target':<10} | {'Max Target':<10} | {'Max Error':<10}")
    print("-" * 100)
    
    lower_body_keywords = ["hip", "knee", "ankle"]
    
    for i, name in enumerate(joint_names):
        # Filter for lower body
        if not any(k in name for k in lower_body_keywords):
            continue

        measured = dof_pos[:, i]
        target = des_dof_pos[:, i]
        error = torch.abs(measured - target)
        
        print(f"{name:<30} | {measured.min():.4f}     | {measured.max():.4f}     | {target.min():.4f}       | {target.max():.4f}       | {error.max():.4f}")

if __name__ == "__main__":
    analyze_latest_data()
