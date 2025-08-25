import os

i = 0
for file in os.listdir("./test_image"):
    if file.endswith(".png"):
        old_path = os.path.join("./test_image", file)
        new_file_name = str(i) + ".png"
        new_path = os.path.join("./test_image", new_file_name)
        os.rename(old_path, new_path)
        i += 1
        print(f"Renamed {file} to {new_file_name}")
        # break  # Remove this if you want to rename all PNG files