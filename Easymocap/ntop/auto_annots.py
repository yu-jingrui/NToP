# AUTOMATED bulk annots.npy file generation

import os
import subprocess
import time
from tqdm import tqdm

root_path = '/path/to/NToP/dataset/'

# Take user inputs
script_choice = input("Enter 'annots' or 'vertices' as the type of result: ").lower()
subject_choice = input("Enter the Subject you want. Enter: S1, S7 and so on, or genebody: ")

data_path = os.path.join(root_path, subject_choice)
subfolders = [f.path for f in os.scandir(data_path) if f.is_dir()]
for subfolder in tqdm(subfolders):
    #print(f'working in {subfolder}')
    if script_choice == 'annots':
        script_name = 'ntop/easymocap_annots.py'
        script_arguments = [
            '--input_dir',subfolder,
            '--type', 'annots'

        ]
    elif script_choice == 'vertices':
        script_name = 'ntop/easymocap_annots.py'
        script_arguments = [
            '--input_dir',subfolder,
            '--type', 'vertices'
        ]
    else:
        print("Invalid script choice. Please choose vertices or annots'.")
        exit(1)

    command = ['python', script_name, *script_arguments]
    subprocess.run(command)
'''
data_path = os.path.join(root_path, subject_choice)
subfolders = [f.path for f in os.scandir(data_path) if f.is_dir()]
if script_choice == 'annots':
    script_name = 'easymocap_annots.py'
    script_arguments = [
        '--input_dir',data_path,
        '--type', 'annots'

    ]
elif script_choice == 'vertices':
    script_name = 'easymocap_annots.py'
    script_arguments = [
        '--input_dir',data_path,
        '--type', 'vertices'
    ]
else:
    print("Invalid script choice. Please choose vertices or annots'.")
    exit(1)

command = ['python', script_name, *script_arguments]
subprocess.run(command)

'''