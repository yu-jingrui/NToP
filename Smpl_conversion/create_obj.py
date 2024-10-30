#h36m_subjects = ['S8', 'S9', 'S11']
h36m_subjects = ['S1']
h36m_actions = ["Directions", "Discussion", "Eating", "Greeting", "Phoning",
                "Photo", "Posing", "Purchases", "Sitting", "SittingDown",
                "Smoking", "Waiting", "WalkDog", "Walking", "WalkTogether"]

import os.path as osp
import argparse

import numpy as np
import torch

import pyrender
import trimesh

import smplx
#from smplx.joint_names import Body

from tqdm.auto import tqdm, trange

from pathlib import Path
import os

from third_parties.smpl.smpl_numpy import SMPL


model_folder = '/home/ndip/smplx/transfer_data/body_models/'

model = smplx.create(
        model_folder,
        model_type='smpl',
        gender='neutral',
        use_face_contour=False,
        num_betas=10,
        num_expression_coeffs=10,
        use_pca=False,
    )

MODEL_DIR = '/home/ndip/humannerf/third_parties/smpl/models'
dataset_dir = '/mnt/data/ndip/h36m_easymocap/'
sex = 'neutral'
smpl_model = SMPL(sex=sex, model_dir=MODEL_DIR)
pbar = tqdm(total=len(h36m_subjects) * len(h36m_actions), desc="Processing Subjects")

for sub in h36m_subjects:
    for ac in h36m_actions:
        subject_dir = os.path.join(dataset_dir, sub, ac)
        pbar.set_description(f"Processing {sub}-{ac}")
        smpl_params_dir = os.path.join(subject_dir,"params")
        #print(f"The directory {smpl_params_dir} {'exists' if os.path.exists(smpl_params_dir) else 'does not exist'}.")
        smpl_dir = os.listdir(smpl_params_dir)
        
        for idx, sm_dir in enumerate(smpl_dir):
            if idx % 5 == 0:
                print(idx)
                try:
                    smpl_params = np.load(os.path.join(smpl_params_dir, sm_dir), allow_pickle=True).item()
                    smpl_params = smpl_params['annots'][0]
                except:
                    smpl_params = np.load(os.path.join(smpl_params_dir, sm_dir), allow_pickle=True).item()
                    smpl_params = smpl_params
        
                betas = np.array(smpl_params['shapes'])[0] #(10,)
                poses = np.array(smpl_params['poses'])[0]  #(72,)
                Rh = np.array(smpl_params['Rh'])[0]  #(3,)
                Th = np.array(smpl_params['Th'])[0]
                vertices, joints = smpl_model(poses, betas)
                #body_pose = poses[3:]
                #for pose_idx in trange(body_pose.size):
                for pose_idx in trange(poses.size):
                    pose_idx = [pose_idx]
                vertices, joints = smpl_model(poses, betas)
                
                vertices = torch.tensor(vertices).float()
                joints = torch.tensor(joints).float()
                vertices = vertices.detach().cpu().numpy().squeeze()
                joints = joints.detach().cpu().numpy().squeeze()
                vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
                tri_mesh = trimesh.Trimesh(
                        vertices, model.faces, vertex_colors=vertex_colors, process=False
                    )
                output_folder = os.path.join(subject_dir, 'obj_files')
                os.makedirs(output_folder, exist_ok=True)
                print(output_folder)
                output_path = f"{output_folder}/{idx:06d}.obj"
                tri_mesh.export(str(output_path))
