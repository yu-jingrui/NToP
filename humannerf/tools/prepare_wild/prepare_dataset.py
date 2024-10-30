import os
import sys

import json
import yaml
import pickle
import numpy as np
from tqdm import tqdm

from pathlib import Path
sys.path.append(str(Path(os.getcwd()).resolve().parents[1]))
from third_parties.smpl.smpl_numpy import SMPL

from absl import app
from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_string('cfg',
                    'wild.yaml',
                    'the path of config file')

MODEL_DIR = '../../third_parties/smpl/models'


def parse_config():
    config = None
    with open(FLAGS.cfg, 'r') as file:
        config = yaml.full_load(file)

    return config


def main(argv):
    del argv  # Unused.

    cfg = parse_config()
    subject = cfg['dataset']['subject']
    sex = cfg['dataset']['sex']

    dataset_dir = cfg['dataset']['path']
    print(dataset_dir, '; exist:', os.path.isdir(dataset_dir))
    subject_dir = os.path.join(dataset_dir, f"Human36m_{subject}")
    output_path = subject_dir
    
    anno_path = os.path.join(subject_dir,'Directions', 'annots.npy')
    print(anno_path, '; exist:', os.path.isfile(anno_path))
    annots = np.load(anno_path, allow_pickle=True).item()
    select_view = cfg['training_view']

    # load cameras
    cams = annots['cams']
    cam_Ks = np.array(cams['K'])[select_view].astype('float32')
    cam_Rs = np.array(cams['R'])[select_view].astype('float32')
    cam_Ts = np.array(cams['T'])[select_view].astype('float32') / 1000.
    cam_Ds = np.array(cams['D'])[select_view].astype('float32')

    K = cam_Ks     #(3, 3)
    K[0,0] = 443.4
    K[1,1] = 443.4
    print(K)
    D = cam_Ds[:, 0]
    E = np.eye(4)  #(4, 4)
#    cam_T = cam_Ts[:3, 0]
#    E[:3, :3] = cam_Rs
#    E[:3, 3]= cam_T
    print(E)

    output_path = os.path.join(cfg['output']['dir'],
                               subject if 'name' not in cfg['output'].keys() else cfg['output']['name'])
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(subject_dir, 'Directions', 'metadata.json'), 'r') as f:
        frame_infos = json.load(f)

    smpl_model = SMPL(sex=sex, model_dir=MODEL_DIR)

    cameras = {}
    mesh_infos = {}
    all_betas = []
    for frame_base_name in tqdm(frame_infos):
        print(frame_base_name)
        cam_body_info = frame_infos[frame_base_name]
        print(cam_body_info)
        poses = np.array(cam_body_info['body_pose'], dtype=np.float32)
        if poses.shape[1] < 72:
            zeros_to_append = 72 - poses.shape[1]
            pose = np.pad(poses, ((0, 0), (0, zeros_to_append)), mode='constant')
#        pose = np.reshape(body_pose_reshaped, (-1))
#        print(pose.shape)
#        print(f" the shape is {frame_base_name}")
        betas = np.array(cam_body_info['smpl_betas'], dtype=np.float32)
#        K = np.array(cam_body_info['cam_intrinsics'], dtype=np.float32)
#        E = np.array(cam_body_info['cam_extrinsics'], dtype=np.float32)
        all_betas.append(betas)

        ##############################################
        # Below we tranfer the global body rotation to camera pose

        # Get T-pose joints
        _, tpose_joints = smpl_model(np.zeros_like(poses), betas)
        # get global Rh, Th
#        print(tpose_joints)
        pelvis_pos = tpose_joints[0].copy()
        Th = pelvis_pos
        Rh = poses[:3].copy()

        # get refined T-pose joints
        tpose_joints = tpose_joints - pelvis_pos[None, :]

        # remove global rotation from body pose
        poses[:3] = 0

        # get posed joints using body poses without global rotation
        _, joints = smpl_model(poses, betas)
        joints = joints - pelvis_pos[None, :]

        mesh_infos[frame_base_name] = {
            'Rh': Rh,
            'Th': Th,
            'poses': poses,
            'joints': joints,
            'tpose_joints': tpose_joints
        }

        cameras[frame_base_name] = {
            'intrinsics': K,
            'extrinsics': E
        }

    # write camera infos
    with open(os.path.join(output_path, 'cameras.pkl'), 'wb') as f:   
        pickle.dump(cameras, f)
        
    # write mesh infos
    with open(os.path.join(output_path, 'mesh_infos.pkl'), 'wb') as f:   
        pickle.dump(mesh_infos, f)

    # write canonical joints
    avg_betas = np.mean(np.stack(all_betas, axis=0), axis=0)
    smpl_model = SMPL(sex, model_dir=MODEL_DIR)
    _, template_joints = smpl_model(np.zeros(72), avg_betas)
    with open(os.path.join(output_path, 'canonical_joints.pkl'), 'wb') as f:   
        pickle.dump(
            {
                'joints': template_joints,
            }, f)


if __name__ == '__main__':
    app.run(main)
