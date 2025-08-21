import h5py
import argparse
import numpy as np


np.set_printoptions(formatter={'float_kind': lambda x: f"{x:.9f}"})

def print_full_h5_structure(h5_file_path):
    with h5py.File(h5_file_path, "r") as f:
        print("📂 Full HDF5 structure:\n")

        def visitor(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"[Group]   {name}")
            elif isinstance(obj, h5py.Dataset):
                try:
                    data = obj[()]
                    flat_data = np.array(data).flatten()
                    size = flat_data.size

                    if size == 1:
                        print(f"[Dataset] {name} (shape={obj.shape}, dtype={obj.dtype}) = {flat_data[0]}")
                    else:
                        print(f"[Dataset] {name} (shape={obj.shape}, dtype={obj.dtype})")
                        for val in flat_data[:5]:
                            print(f"  {val}")
                        if size > 5:
                            print(f"  ... ({size - 5} more)")
                except Exception as e:
                    print(f"[Dataset] {name} (shape={obj.shape}, dtype={obj.dtype})")
                    print(f"  ERROR reading dataset: {e}")
            else:
                print(f"[Unknown] {name} ¯\\_(ツ)_/¯")

        f.visititems(visitor)



if __name__ == '__main__':
  # Argument parser
  parser = argparse.ArgumentParser(description="convert rosbag2 to source")
  parser.add_argument(
    "--h5file",
    required=True,
    help="Path to the h5 file to be inspected",
  )
  args = parser.parse_args()

  print_full_h5_structure(args.h5file)