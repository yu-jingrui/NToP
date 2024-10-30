import conversion
import pickle
import json
import os
from pathlib import Path
import numpy as np

# Define labels for COCO and SMPL
coco_labels = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle"
]

smpl_labels = [
    'MidHip',            # 0
    'LUpLeg',       # 1
    'RUpLeg',      # 2
    'spine',           # 3
    'LLeg',         # 4
    'RLeg',        # 5
    'spine1',          # 6
    'LFoot',        # 7
    'RFoot',       # 8
    'spine2',          # 9
    'LToeBase',     # 10
    'RToeBase',    # 11
    'neck',            # 12
    'LShoulder',    # 13
    'RShoulder',   # 14
    'head',            # 15
    'LArm',         # 16
    'RArm',        # 17
    'LForeArm',     # 18
    'RForeArm',    # 19
    'LHand',        # 20
    'RHand',       # 21
    'LHandIndex1',  # 22
    'RHandIndex1', 
]

def extract_coco(joints_3d, extrinsic, intrinsic, height, width):
    # Use the conversion function from your 'conversion' library to convert 3D to COCO 2D
    focal = [intrinsic[0, 0], intrinsic[1, 1]]
    center = (intrinsic[0, 2], intrinsic[1, 2])
    coco_2d = conversion.skeleton3d_to_2d(conversion.coco_convert(joints_3d), extrinsic, height, width, focal, center)
    return coco_2d

def extract_smpl(joints_3d, extrinsic, intrinsic, height, width):
    # Use the conversion function from your 'conversion' library to convert 3D to SMPL 2D
    focal = [intrinsic[0, 0], intrinsic[1, 1]]
    center = (intrinsic[0, 2], intrinsic[1, 2])
    smpl_2d = conversion.skeleton3d_to_2d(joints_3d, extrinsic, height, width, focal, center)
    return smpl_2d

    
def pickle_read(path):
    assert os.path.exists(path), path
    with open(path, 'rb') as f:
        try:
            data = pickle.load(f)
        except:
            print('Reading error {}'.format(path))
            data = []
    return data

def convert_numpy_arrays(data):
    # Convert NumPy arrays to Python lists and convert float32 to regular float
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.float32):
        return float(data)
    elif isinstance(data, dict):
        return {key: convert_numpy_arrays(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_arrays(item) for item in data]
    else:
        return data


def add_confidence_scores(keypoints_2d):
    # Add confidence score of 1 to each keypoint in the 2D keypoints
    return [{'x': pt[0], 'y': pt[1], 'score': 1.0} for pt in keypoints_2d]
    
def add_confidence_to_joints_3d(joints_3d):
    # Flatten the nested list of 3D joint coordinates and add confidence score of 1.0 to each 3D joint coordinate
    flattened_joints = [coord for frame_joints in joints_3d for coord in frame_joints]
    return [{'x': joint[0], 'y': joint[1], 'z': joint[2], 'score': 1.0} for joint in flattened_joints]

def run_single_frame(path):
    data = pickle_read(path)
    combined_data = []

    for frame_number, frame_data in data.items():
        joints_3d = frame_data['joints']
        extrinsic = frame_data['extrinsic']
        intrinsic = frame_data['intrinsic']
        height = frame_data['height']
        width = frame_data['width']
        
        # Extract COCO 2D keypoints
        coco_2d = extract_coco(joints_3d, extrinsic, intrinsic, height, width)

        # Extract SMPL 2D keypoints
        smpl_2d = extract_smpl(joints_3d, extrinsic, intrinsic, height, width)

        # Add confidence scores to the 2D keypoints
        coco_2d_with_confidence = add_confidence_scores(coco_2d)
        smpl_2d_with_confidence = add_confidence_scores(smpl_2d)
        
        joints_3d_with_confidence = add_confidence_to_joints_3d(joints_3d)
        
        # Combine all the data into a single dictionary
        combined_data.append({
            'frame_number': frame_number,
            'coco_2d': convert_numpy_arrays(coco_2d_with_confidence),
            'smpl_2d': convert_numpy_arrays(smpl_2d_with_confidence),
            'joints_3d': convert_numpy_arrays(joints_3d_with_confidence),
            'H': height,
            'W': width,
            # Add any other relevant information here
        })

    return combined_data

h36m_subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
h36m_actions = ["Directions", "Discussion", "Eating", "Greeting", "Phoning",
                "Photo", "Posing", "Purchases", "Sitting", "SittingDown",
                "Smoking", "Waiting", "WalkDog", "Walking", "WalkTogether"]

if __name__ == "__main__":
    render_path = Path('/home/yjin/NTOP_h36m')
    for sub in h36m_subjects:
        for ac in h36m_actions:
            annotations = {}
            sub_ac_path = Path(render_path, sub, ac)
            frames = sorted(os.listdir(sub_ac_path))
            for fr in frames:
                pkl_path = Path(sub_ac_path, fr, 'extrinsic_intrinsic_joints.pkl')
                combined_data = run_single_frame(pkl_path)
                combined_data = convert_numpy_arrays(combined_data)
                annotations[fr] = combined_data
            output_path = Path(sub_ac_path, f'{sub}_{ac}_anno.json')
            with open(output_path, 'w') as json_file:
                json.dump(annotations, json_file, indent=4)
