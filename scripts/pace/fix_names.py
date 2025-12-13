import torch
import sys
import shutil
from pathlib import Path

def fix_joint_names(file_path):
    path = Path(file_path)
    
    if not path.exists():
        print(f"Error: File '{file_path}' does not exist.")
        return

    # 1. Create a backup just in case
    backup_path = path.with_name(f"{path.stem}_backup{path.suffix}")
    if not backup_path.exists():
        shutil.copy(path, backup_path)
        print(f"Created backup at: {backup_path}")

    # 2. Load Data
    print(f"Loading {path}...")
    data = torch.load(path)

    # 3. Define mapping
    name_map = {
        'foot_0_pitch': 'ankle_pitch_left',
        'foot_0_roll':  'ankle_roll_left'
    }

    # 4. Modify
    if 'joint_names' in data:
        old_names = data['joint_names']
        # Replace names if they exist in the map, otherwise keep original
        new_names = [name_map.get(name, name) for name in old_names]
        
        if new_names != old_names:
            data['joint_names'] = new_names
            # Save back to the same file
            torch.save(data, path)
            print("Success! Renamed joints:")
            print(f"  From: {old_names}")
            print(f"  To:   {new_names}")
        else:
            print("No changes needed (names already match or aren't in the map).")
    else:
        print("Key 'joint_names' not found in the dictionary.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 fix_names.py path/to/chirp_data.pt")
    else:
        fix_joint_names(sys.argv[1])