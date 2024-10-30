import os
import sys
import json
from shutil import copyfile

import pickle
import yaml
import numpy as np
from tqdm import tqdm

from pathlib import Path
humannerf_root = Path(os.getcwd()).resolve().parents[1]
sys.path.append(str(humannerf_root))

from third_parties.smpl.smpl_numpy import SMPL
from core.utils.file_util import split_path
from core.utils.image_util import load_image, save_image, to_3ch_image

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('cfg',
                    'h36m.yaml',
                    'the path of config file')

MODEL_DIR = os.path.join(humannerf_root, 'third_parties', 'smpl', 'models')


camera_list = ['54138969', '55011271', '58860488', '60457274']
camera_mapping = {camera: index for index, camera in enumerate(camera_list)} 

def parse_config():
    config = None
    with open(FLAGS.cfg, 'r') as file:
        config = yaml.full_load(file)

    return config


def prepare_dir(output_path, name):
    out_dir = os.path.join(output_path, name)
    os.makedirs(out_dir, exist_ok=True)

    return out_dir


def get_mask(subject_dir, img_name):
    msk_path = os.path.join(subject_dir, 'mask',
                            img_name)[:-4] + '.png'
    msk = np.array(load_image(msk_path))[:, :, 0]
    msk = (msk != 0).astype(np.uint8)

    msk_path = os.path.join(subject_dir, 'mask_cihp',
                            img_name)[:-4] + '.png'
    msk_cihp = np.array(load_image(msk_path))[:, :, 0]
    msk_cihp = (msk_cihp != 0).astype(np.uint8)

    msk = (msk | msk_cihp).astype(np.uint8)
    msk[msk == 1] = 255

    return msk


def main(argv):
    del argv  # Unused.

    cfg = parse_config()
    subject = cfg['dataset']['subject']
    action = cfg['dataset']['action']
    sex = cfg['dataset']['sex']
    max_frames = cfg['max_frames']

    dataset_dir = cfg['dataset']['h36m_path']
    print(dataset_dir, '; exist:', os.path.isdir(dataset_dir))
    subject_dir = os.path.join(dataset_dir, f"{subject}")
    
    smpl_params_dir = os.path.join(subject_dir, action, "params")
    #smpl_params_dir = os.path.join(subject_dir, 'Posing_easymocap', "params")
    print(smpl_params_dir, '; exist:', os.path.isdir(smpl_params_dir))
    
    anno_path = os.path.join(subject_dir, action, 'annots.npy')
    #anno_path = os.path.join(subject_dir, 'Posing_easymocap', 'annots.npy')
    print(anno_path, '; exist:', os.path.isfile(anno_path))
    annots = np.load(anno_path, allow_pickle=True).item()
    
    cams = annots['cams']
    Ks = []
    Ds = []
    Es = []
    for i in range(4):
        cam_Ks = np.array(cams['K'])[i].astype('float32')
        cam_Rs = np.array(cams['R'])[i].astype('float32')
        cam_Ts = np.array(cams['T'])[i].astype('float32') / 1000.
        cam_Ds = np.array(cams['D'])[i].astype('float32')

        K = cam_Ks     #(3, 3)
        D = cam_Ds[:, 0]
        E = np.eye(4)  #(4, 4)
        cam_T = cam_Ts[:3, 0]
        E[:3, :3] = cam_Rs
        E[:3, 3]= cam_T

        Ks.append(K)
        Ds.append(D)
        Es.append(E)
     
    # load image paths
    img_path_frames_views = annots['ims']

    #img_path = []
    #img_paths = []
    #for i in range(4):
    #    img_path.append([multi_view_paths['ims'][i] for multi_view_paths in img_path_frames_views])
    #    img_path[i] = img_path[i][i:max_frames:4]
    #    img_paths = img_paths + img_path[i] 
    #img_paths = np.array(img_paths)
    
    img_path_frames_views = img_path_frames_views[::5]
    img_paths = []
    for multi_view_paths in img_path_frames_views:
        img_paths += multi_view_paths['ims']
    
    output_path = os.path.join(cfg['output']['dir'],
                       subject if 'name' not in cfg['output'].keys() else cfg['output']['name'])
    os.makedirs(output_path, exist_ok=True)
    out_img_dir  = prepare_dir(output_path, 'images')
    out_mask_dir = prepare_dir(output_path, 'masks')
    
    copyfile(FLAGS.cfg, os.path.join(output_path, 'config.yaml'))

    smpl_model = SMPL(sex=sex, model_dir=MODEL_DIR)
    
    cameras = {}
    mesh_infos = {}
    all_betas = []

    running_idx = 0
    for ipath in tqdm(img_paths):
        print(ipath)
        idx = int(ipath[16:].rstrip('.jpg'))
#        if idx % 5 != 0:
#            continue
        print(f"idx is {idx}")
        out_name = f'frame_{running_idx:06d}'

        img_path = os.path.join(subject_dir, action, ipath)
        print(img_path)
        # load image
        img = np.array(load_image(img_path))

        smpl_idx = idx
        smpl_params_file = os.path.join(smpl_params_dir, f"{smpl_idx:06d}.npy")
        print(smpl_params_file)
        try:
            smpl_params = np.load(smpl_params_file, allow_pickle=True).item()
            smpl_params = smpl_params['annots'][0]
            #smpl_params = json.load(smpl_params_file).item()
        except:
            #continue
            raise FileNotFoundError(f'SMPL file {smpl_params_file} not found.')
        betas = np.array(smpl_params['shapes'])[0] #(10,)
        poses = np.array(smpl_params['poses'])[0]  #(72,)
        Rh = np.array(smpl_params['Rh'])[0]  #(3,)
        Th = np.array(smpl_params['Th'])[0]  #(3,)

        all_betas.append(betas)
        
        # write camera info
        cam_number = ipath[:8]
        if cam_number == camera_list[0]:
            cameras[out_name] = {
                    'intrinsics': Ks[0],
                    'extrinsics': Es[0],
                    'distortions': Ds[0]
            }
        elif cam_number == camera_list[1]:
            cameras[out_name] = {
                    'intrinsics': Ks[1],
                    'extrinsics': Es[1],
                    'distortions': Ds[1]
            }
        elif cam_number == camera_list[2]:
            cameras[out_name] = {
                    'intrinsics': Ks[2],
                    'extrinsics': Es[2],
                    'distortions': Ds[2]
            }
        else:
            cameras[out_name] = {
                    'intrinsics': Ks[3],
                    'extrinsics': Es[3],
                    'distortions': Ds[3]
            }
        
        # write mesh info
        _, tpose_joints = smpl_model(np.zeros_like(poses), betas)
        _, joints = smpl_model(poses, betas)
        mesh_infos[out_name] = {
            'Rh': Rh,
            'Th': Th,
            'poses': poses,
            'joints': joints,
            'tpose_joints': tpose_joints
        }

        # load and write mask
        #mask = get_mask(os.path.join(subject_dir, action), ipath)
        #save_image(to_3ch_image(mask), os.path.join(out_mask_dir, out_name+'.png'))

        # write image
        out_image_path = os.path.join(out_img_dir, '{}.png'.format(out_name))
        save_image(img, out_image_path)
        
        running_idx += 1
    
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
