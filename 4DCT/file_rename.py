import os

path = "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/data/artificial/artificial_N10_S10000_1000/image/"

# Iterate over the folders in the specified path
for folder_name in os.listdir(path):
    folder_path = os.path.join(path, folder_name)
    
    # Check if the item in the path is a folder
    if os.path.isdir(folder_path):
        # Extract the current folder name as an integer
        folder_index = int(folder_name)
        
        # Create the new folder name with three digits
        new_folder_name = str(folder_index).zfill(3)
        
        # Create the new path with the updated folder name
        new_folder_path = os.path.join(path, new_folder_name)
        
        # Rename the folder
        os.rename(folder_path, new_folder_path)