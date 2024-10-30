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
    out_dir = Path(output_path, name)
    os.makedirs(out_dir, exist_ok=True)

    return out_dir


def get_mask(subject_dir, img_name):
    msk_path = Path(subject_dir, 'mask',
                            img_name)[:-4] + '.png'
    msk = np.array(load_image(msk_path))[:, :, 0]
    msk = (msk != 0).astype(np.uint8)

    msk_path = Path(subject_dir, 'mask_cihp',
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
    actions = cfg['dataset']['actions']
    sex = cfg['dataset']['sex']
    view = cfg['view']
    max_frames = cfg['max_frames']

    dataset_dir = cfg['dataset']['h36m_path']
    print(dataset_dir, '; exist:', os.path.isdir(dataset_dir))
    subject_dir = Path(dataset_dir, f"{subject}")
    
    for ac in actions:
        smpl_params_dir = Path(subject_dir, ac, "params")
        #smpl_params_dir = Path(subject_dir, 'Posing_easymocap', "params")
        print(smpl_params_dir, '; exist:', os.path.isdir(smpl_params_dir))

        anno_path = Path(subject_dir, ac, 'annots.npy')
        #anno_path = Path(subject_dir, 'Posing_easymocap', 'annots.npy')
        print(anno_path, '; exist:', os.path.isfile(anno_path))
        annots = np.load(anno_path, allow_pickle=True).item()
        
        cam_anno_path = Path('/mnt/data/yjin/Human36m_S8/Posing/annots.npy')
        cam_annots = np.load(cam_anno_path, allow_pickle=True).item()
        cams = cam_annots['cams']
        cam_idx = camera_mapping[view]
        
        cam_Ks = np.array(cams['K'])[cam_idx].astype('float32')
        cam_Rs = np.array(cams['R'])[cam_idx].astype('float32')
        cam_Ts = np.array(cams['T'])[cam_idx].astype('float32') / 1000.
        cam_Ds = np.array(cams['D'])[cam_idx].astype('float32')

        K = cam_Ks     #(3, 3)
        D = cam_Ds[:, 0]
        E = np.eye(4)  #(4, 4)
        cam_T = cam_Ts[:3, 0]
        E[:3, :3] = cam_Rs
        E[:3, 3]= cam_T

        # load image paths
        img_paths = [item['ims'][cam_idx] for item in annots['ims']]

        output_path = Path(cfg['output']['dir'], subject, ac)
        os.makedirs(output_path, exist_ok=True)
        out_img_dir  = prepare_dir(output_path, 'images')
        out_mask_dir = prepare_dir(output_path, 'masks')

        copyfile(FLAGS.cfg, Path(output_path, 'config.yaml'))

        smpl_model = SMPL(sex=sex, model_dir=MODEL_DIR)

        cameras = {}
        mesh_infos = {}
        all_betas = []
        
        #betas_file_path = Path('/mnt/data/yjin/Human36m_S8/Posing/new_params/0.npy')
        #betas_params = np.load(betas_file_path, allow_pickle=True).item()
        #betas = np.array(betas_params['shapes'])[0]

        running_idx = 0
        for ipath in tqdm(img_paths):
            idx = ipath[-10:-4]
            if int(idx) % 5 != 0:
                continue
            out_name = idx

            img_path = Path(subject_dir, ac, ipath)
            # load image
            img = np.array(load_image(img_path))

            smpl_params_file = Path(smpl_params_dir, f"{idx}.npy")
            try:
                smpl_params = np.load(smpl_params_file, allow_pickle=True).item()
                #smpl_params = smpl_params['annots'][0]
                #smpl_params = json.load(smpl_params_file).item()
            except:
                #continue
                raise FileNotFoundError(f'SMPL file {smpl_params_file} not found.')

            #smpl_params = smpl_params['annots'][0]

            betas = np.array(smpl_params['shapes'])[0] #(10,)
            poses = np.array(smpl_params['poses'])[0]  #(72,)
            Rh = np.array(smpl_params['Rh'])[0]  #(3,)
            Th = np.array(smpl_params['Th'])[0]  #(3,)

            all_betas.append(betas)

            # write camera info
            cameras[out_name] = {
                    'intrinsics': K,
                    'extrinsics': E,
                    'distortions': D
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
            #mask = get_mask(Path(subject_dir, action), ipath)
            #save_image(to_3ch_image(mask), Path(out_mask_dir, out_name+'.png'))

            # write image
            out_image_path = Path(out_img_dir, '{}.png'.format(out_name))
            save_image(img, out_image_path)

            running_idx += 1

        # write camera infos
        with open(Path(output_path, 'cameras.pkl'), 'wb') as f:
            pickle.dump(cameras, f)

        # write mesh infos
        with open(Path(output_path, 'mesh_infos.pkl'), 'wb') as f:
            pickle.dump(mesh_infos, f)

        # write canonical joints
        avg_betas = np.mean(np.stack(all_betas, axis=0), axis=0)
        smpl_model = SMPL(sex, model_dir=MODEL_DIR)
        _, template_joints = smpl_model(np.zeros(72), avg_betas)
        with open(Path(output_path, 'canonical_joints.pkl'), 'wb') as f:
            pickle.dump(
                {
                    'joints': template_joints,
                }, f)

if __name__ == '__main__':
    app.run(main)
