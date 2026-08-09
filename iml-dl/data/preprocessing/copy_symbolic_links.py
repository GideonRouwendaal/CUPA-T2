"""
Script to copy the recon_test folder with symbolic links and create an "artificial"
recon_test-motionfree_motion folder to test the model's performance on motion-free data.
"""

import os
import shutil

def copy_folder_structure_and_symlinks(src_folder, dest_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Traverse the source folder
    for root, dirs, files in os.walk(src_folder):
        # Calculate the relative path from the source folder
        relative_path = os.path.relpath(root, src_folder)
        current_dest_path = os.path.join(dest_folder, relative_path)

        # Create the corresponding folder in the destination
        os.makedirs(current_dest_path, exist_ok=True)

        # Handle symbolic links
        for file in files:
            src_file_path = os.path.join(root, file)
            dest_file_path = os.path.join(current_dest_path, file).replace("_nr_", "-motionfree_nr_")

            # If the file is a symbolic link
            if os.path.islink(src_file_path):
                symlink_target = os.readlink(src_file_path)
                os.symlink(symlink_target, dest_file_path)

                renamed_symlink_path = dest_file_path.replace("fV4", "f_moveV4")  # Rename the link
                os.symlink(symlink_target, renamed_symlink_path)  # Create renamed symlink


if __name__ == "__main__":
    src_folder = "./data/links_to_data/recon_test/"
    dest_folder = "./data/links_to_data/recon_test-motionfree_motion/"

    copy_folder_structure_and_symlinks(src_folder, dest_folder)
