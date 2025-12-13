import torch
import os
from pathlib import Path

def project_root():
    # Adjust this index if your file location changes (parents[2] moves up 3 levels)
    return Path(__file__).resolve().parents[2]

def analyze_latest_data():
    data_root = project_root() / "data" / "hoku_flat_left_ankle_sim"
    
    # Check if data root exists
    if not data_root.exists():
         print(f"Data root {data_root} does not exist.")
         return

    # Find latest directory
    subdirs = [d for d in data_root.iterdir() if d.is_dir()]
    if not subdirs:
        print("No data directories found.")
        return
    
    latest_dir = max(subdirs, key=os.path.getmtime)
    print(f"Analyzing data from: {latest_dir}")
    print("=" * 60)
    
    data_path = latest_dir / "chirp_data.pt"
    if not data_path.exists():
        print(f"No chirp_data.pt found in {latest_dir}")
        return

    try:
        # Load the dictionary
        data = torch.load(data_path, map_location='cpu')
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    print(f"Loaded {data_path.name}")
    print(f"Type of loaded object: {type(data)}")
    print("-" * 60)

    # Handle if the loaded data is a dictionary
    if isinstance(data, dict):
        for key, value in data.items():
            print_value_info(key, value)
            print("-" * 40)
    
    # Handle if it's just a Tensor
    elif torch.is_tensor(data):
        print_value_info("Single Tensor", data)
    
    # Handle lists/tuples
    elif isinstance(data, (list, tuple)):
        print(f"Data is a {type(data).__name__} with {len(data)} elements.")
        for i, value in enumerate(data[:5]): # Print info for first 5 elements
            print_value_info(f"Index {i}", value)
            print("-" * 40)
        if len(data) > 5:
            print("... (remaining elements omitted)")
    
    else:
        print(f"Data structure not explicitly handled: {data}")

def print_value_info(name, value):
    """Helper to print type, shape, and first 10 values."""
    print(f"Key/Index: '{name}'")
    print(f"  Type: {type(value).__name__}")
    
    # If it's a Tensor
    if torch.is_tensor(value):
        print(f"  Shape: {list(value.shape)}")
        print(f"  Dtype: {value.dtype}")
        print(f"  Device: {value.device}")
        
        # Flatten to print first 10 elements easily
        flat_vals = value.flatten()
        preview = flat_vals[:10].tolist()
        print(f"  First 10 values (flattened): {preview}")
        
    # If it's a Numpy array
    elif hasattr(value, 'shape') and hasattr(value, 'flatten'):
        print(f"  Shape: {value.shape}")
        flat_vals = value.flatten()
        # Handle large arrays safely
        preview = flat_vals[:10] if len(flat_vals) > 0 else []
        print(f"  First 10 values (flattened): {preview}")

    # If it's a list or tuple
    elif isinstance(value, (list, tuple)):
        print(f"  Length: {len(value)}")
        print(f"  First 10 items: {value[:10]}")
        
    # Basic scalar types
    else:
        print(f"  Value: {value}")

if __name__ == "__main__":
    analyze_latest_data()