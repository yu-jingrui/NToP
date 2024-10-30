import os
import sys

from shutil import copyfile

import pickle
import yaml
import numpy as np
from tqdm import tqdm

from pathlib import Path
from natsort import natsorted
humannerf_root = Path(os.getcwd()).resolve().parents[1]
sys.path.append(str(humannerf_root))

from third_parties.smpl.smpl_numpy import SMPL
from core.utils.file_util import split_path
from core.utils.image_util import load_image, save_image, to_3ch_image

from absl import app
from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_string('cfg',
                    '387.yaml',
                    'the path of config file')

MODEL_DIR = os.path.join(humannerf_root, 'third_parties', 'smpl', 'models')

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
    sex = cfg['dataset']['sex']
    max_frames = cfg['max_frames']

    dataset_dir = cfg['dataset']['genebody_path']
    subject_dir = os.path.join(dataset_dir, subject)
    smpl_params_dir = os.path.join(subject_dir,"params")


    select_view = cfg['training_view']

    anno_path = os.path.join(subject_dir, 'annots.npy')
    annots = np.load(anno_path, allow_pickle=True).item()
    
    # load cameras
    cams = annots['cams']
        #joints = np.array([joint - pelvis for joint in joints])
    Ks = {}
    Ds = {}
    Es = {}
    for cam_idx, sc in enumerate(select_view):
        cam_Ks = np.array(cams['K'])[cam_idx].astype('float32')
        cam_Rs = np.array(cams['R'])[cam_idx].astype('float32')
        cam_Ts = np.array(cams['T'])[cam_idx].astype('float32') / 1000.
        cam_Ds = np.array(cams['D'])[cam_idx].astype('float32')

        
        K = cam_Ks     #(3, 3)

        D = np.array([cam_Ds])
        E = np.eye(4)  #(4, 4)

        cam_T = cam_Ts[:3, 0]

        E[:3, :3] = cam_Rs 
        E[:3, 3]= cam_T
        #E = np.linalg.inv(E)
        '''
        f_x_original = K[0,0]
        f_y_original = K[1,1]
        c_x_original = K[0,2]
        c_y_original = K[1,2]

        scale_x = 1000 / 2448
        scale_y = 1000 / 2048

        f_x_new = f_x_original * scale_x
        f_y_new = f_y_original * scale_y

        c_x_new = c_x_original * scale_x
        c_y_new = c_y_original * scale_y
        
        #K[0,0] = f_x_new
        #K[1,1] = f_y_new
        K[0,2] = c_x_new
        K[1,2] = c_y_new
        '''
        Ks[sc] = K
        Ds[sc] = D
        Es[sc] = E

        
    # load image paths
    img_filenames = natsorted(os.listdir(Path(subject_dir, 'images/01')))
    img_paths = []
    
    for f in img_filenames:
        for sc in select_view:
            img_paths.append(sc + '/' + f)

    output_path = os.path.join(cfg['output']['dir'], 
                               subject if 'name' not in cfg['output'].keys() else cfg['output']['name'])
    os.makedirs(output_path, exist_ok=True)
    out_img_dir  = prepare_dir(output_path, 'images')
    out_mask_dir = prepare_dir(output_path, 'masks')

    # copy config file
    copyfile(FLAGS.cfg, os.path.join(output_path, 'config.yaml'))

    smpl_model = SMPL(sex=sex, model_dir=MODEL_DIR)

    cameras = {}
    mesh_infos = {}
    all_betas = []
    for idx, ipath in enumerate(tqdm(img_paths)):
        out_name = f'frame_{idx:06d}'
        img_path = os.path.join(subject_dir,'images', ipath)
        #print(img_path)
        # load image
        img = np.array(load_image(img_path))
        
        smpl_idx = ipath[3:].rstrip('.jpg')
        #print(smpl_idx)

        #smpl_idx = idx

        smpl_params_file = os.path.join(smpl_params_dir, f"{smpl_idx}.npy")
        #print(smpl_params_file)
        try:
            smpl_params = np.load(smpl_params_file, allow_pickle=True).item()
            smpl_params = smpl_params['annots'][0]
        except:
            raise FileNotFoundError(f'SMPL file {smpl_params_file} not found.')
        betas = np.array(smpl_params['shapes'])[0] #(10,)
        poses = np.array(smpl_params['poses'])[0]  #(72,)
        Rh = np.array(smpl_params['Rh'])[0]  #(3,)
        Th = np.array(smpl_params['Th'])[0]
        '''
        betas = smpl_params['betas'][0] #(10,)
        Rh = smpl_params['global_orient'][0]  #(3,)
        Th = smpl_params['transl'][0]  #(3,)
        poses = kp.reshape(72,)  #(72,)
        '''
        all_betas.append(betas)

        # write camera info
        cam_number = ipath[:2]

        cameras[out_name] = {
                'intrinsics': Ks[cam_number],
                'extrinsics': Es[cam_number],
                'distortions': Ds[cam_number]
        }

        # write mesh info
        _, tpose_joints = smpl_model(np.zeros_like(poses), betas)
        _, joints = smpl_model(poses, betas)
        
        joints[:,0] = -joints[:,0]
        
        mesh_infos[out_name] = {
            'Rh': Rh,
            'Th': Th,
            'poses': poses,
            'joints': joints, 
            'tpose_joints': tpose_joints
        }

        '''
        # load and write mask
        mask = get_mask(subject_dir, ipath)
        save_image(to_3ch_image(mask), 
                   os.path.join(out_mask_dir, out_name+'.png'))
        '''

        mpath = ipath[:3] + 'mask' + ipath[3:].rstrip('jpg') + 'png'
        #print(mpath)
        copyfile(Path(subject_dir, 'masks', mpath), Path(out_mask_dir, f'{out_name}.png'))

        # write image
        out_image_path = os.path.join(out_img_dir, '{}.png'.format(out_name))
        save_image(img, out_image_path)
        

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
