import os
import subprocess
import time
from tqdm import tqdm

#root_path = '/path/to/NToP/dataset/h36m' # pre-processed h36m folder
#subjects = ['S1'] # subjects that you want to process
root_path = '/path/to/NToP/dataset/' # dataset root folder
subjects = ['genebody'] # this should be the genebody root folder, not the subjects in genebody

# Take user inputs
script_choice = input("Enter 'mocap' to use mocap.py or 'mv1p' to use mv1p.py: ").lower()
#mv1p_subjects = ['S1', 'S5', 'S7']
#gender = 'female' if any(sub in subjects for sub in mv1p_subjects) else 'male'

# camera file numbers, first line for h36m, second line for genebody
#sub_values = ('54138969', '55011271', '58860488', '60457274') 
sub_values = ('01', '10', '19', '28', '37', '46')

for sub in subjects:
    data_path = os.path.join(root_path, sub)

    subfolders = [f.path for f in os.scandir(data_path) if f.is_dir()]

    for subfolder in tqdm(subfolders):

        output_folder = os.path.join(subfolder, 'output')
        os.makedirs(output_folder, exist_ok=True)

        if script_choice == 'mocap':
            script_name = 'mocap.py'
            script_arguments = [
                '--work', 'mv1p',
                '--subs', *sub_values,
    #                '--subs_vis', *sub_values,  # to visualize smpl per camera
                '--mode', 'smplh-3d'

            ]
        elif script_choice == 'mv1p':    #multi-view single person
            script_name = 'mv1p.py'
            script_arguments = [
                '--sub', *sub_values,
                '--body', 'body25',
                '--gender', 'neutral',
                '--out', output_folder,
                '--model', 'smpl',
                '--sub_vis', *sub_values,
#                '--vis_smpl',
                '--write_smpl_full'
            ]
        else:
            print("Invalid script choice. Please choose 'mocap' or 'mv1p'.")
            break

        # Start the timer
        start_time = time.time()

        # Build the command to execute the chosen script with arguments
        command = ['python', 'apps/demo/' + script_name, subfolder, *script_arguments]

        # Execute the command
        subprocess.run(command)

        # End the timer
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Save elapsed time for this subfolder in the subject's file
        # subject_elapsed_filename = os.path.join(data_path, 'elapsed_times.txt')
        # with open(subject_elapsed_filename, 'a') as subject_elapsed_file:
        #     subject_elapsed_file.write(f"Subfolder: {os.path.basename(data_path)}\n")
        #     subject_elapsed_file.write(f"Elapsed time: {elapsed_time:.2f} seconds\n\n")

        # print(f"Elapsed time saved in {subject_elapsed_filename}\n")
