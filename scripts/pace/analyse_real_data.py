import torch
import argparse
from pathlib import Path
import sys

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
        preview = flat_vals[:10] if len(flat_vals) > 0 else []
        print(f"  First 10 values (flattened): {preview}")

    # If it's a list or tuple
    elif isinstance(value, (list, tuple)):
        print(f"  Length: {len(value)}")
        print(f"  First 10 items: {value[:10]}")
        
    # Basic scalar types
    else:
        print(f"  Value: {value}")

def analyze_data(input_path_str):
    input_path = Path(input_path_str).resolve()
    
    # Logic to find the actual .pt file
    file_to_load = None

    if input_path.is_file():
        file_to_load = input_path
    elif input_path.is_dir():
        # Check for standard name inside directory
        candidate = input_path / "chirp_data.pt"
        if candidate.exists():
            file_to_load = candidate
    elif not input_path.exists():
        # Handle case where user provided filename without extension (e.g. .../chirp_data)
        candidate = input_path.with_suffix(".pt")
        if candidate.exists():
            file_to_load = candidate

    if file_to_load is None or not file_to_load.exists():
        print(f"Error: Could not locate a valid data file at or inside: {input_path}")
        return

    print(f"Analyzing data from: {file_to_load}")
    print("=" * 60)

    try:
        data = torch.load(file_to_load, map_location='cpu')
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    print(f"Loaded {file_to_load.name}")
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
        for i, value in enumerate(data[:5]): 
            print_value_info(f"Index {i}", value)
            print("-" * 40)
    
    else:
        print(f"Data structure not explicitly handled: {data}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze contents of a PyTorch .pt file.")
    parser.add_argument("path", type=str, help="Path to the .pt file or directory containing chirp_data.pt")
    
    args = parser.parse_args()
    analyze_data(args.path)