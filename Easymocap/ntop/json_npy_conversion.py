# Specify the source folder path where the .json files are located
#source_folder = '/mnt/data/ndip/h36m_easymocap/S6/WalkTogether/output/smpl_full'
#source_folder = '/mnt/data/ndip/genebody/gaoxing/output-output-smpl-3d/smplfull'


# Specify the destination folder path where you want to save the .npy files
#destination_folder = '/mnt/data/ndip/h36m_easymocap/S6/WalkTogether/params'
#destination_folder = '/mnt/data/ndip/genebody/gaoxing/params'

import os
import numpy as np
import json

def convert_json_to_npy(subject):
    # Specify the source and destination base folders
    source_base_folder = f'/path/to/NToP/dataset/{subject}'
    destination_base_folder = f'/path/to/NToP/dataset/{subject}'

    # Iterate through all subdirectories in the source folder
    for subdirectory in os.listdir(source_base_folder):
        source_folder = os.path.join(source_base_folder, subdirectory, 'output-output-smpl-3d/smplfull')
        destination_folder = os.path.join(destination_base_folder, subdirectory, 'params')

        # List all files in the source folder
        files = os.listdir(source_folder)

        # Filter files to include only .json files
        json_files = [file for file in files if file.endswith('.json')]

        # Create the destination folder if it doesn't exist
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Loop through the .json files, read their contents, and save as .npy files
        for index, json_file in enumerate(json_files):
            source_file_path = os.path.join(source_folder, json_file)

            # Read the contents of the .json file
            try:
                with open(source_file_path, 'r') as json_file:
                    data = json.load(json_file)
            except Exception as e:
                print(f'Error reading {json_file}: {e}')
                continue

            # Create the new filename with leading zeros
            new_index = index
            new_filename = f'{new_index:06d}.npy'

            destination_file_path = os.path.join(destination_folder, new_filename)

            # Save the data as a new .npy file in the destination folder
            try:
                np.save(destination_file_path, data)

            except Exception as e:
                print(f'Error saving {new_filename}: {e}')

        print(f'File conversion process completed for {subject}/{subdirectory}')


# Get the subject as user input
subject = input("Enter the subject (e.g., S7): ")

# Call the conversion function with the specified subject
convert_json_to_npy(subject)

