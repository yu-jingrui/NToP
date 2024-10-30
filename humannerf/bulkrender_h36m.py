import os

import pickle
import torch
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path

from core.data import create_dataloader
from core.nets import create_network
from core.utils.train_util import cpu_data_to_gpu
from core.utils.image_util import ImageWriter, to_8b_image, to_8b3ch_image
from core.data.human_nerf.topview import Dataset

from configs import cfg, args

EXCLUDE_KEYS_TO_GPU = ['frame_name',
                       'img_width', 'img_height', 'ray_mask']


def load_network():
    model = create_network()
    ckpt_path = os.path.join(cfg.logdir, f'{cfg.load_net}.tar')
    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    model.load_state_dict(ckpt['network'], strict=False)
    print('load network from ', ckpt_path)
    return model.cuda().deploy_mlps_to_secondary_gpus()

def unpack_alpha_map(alpha_vals, ray_mask, width, height):
    alpha_map = np.zeros((height * width), dtype='float32')
    alpha_map[ray_mask] = alpha_vals
    return alpha_map.reshape((height, width))


def unpack_to_image(width, height, ray_mask, bgcolor,
                    rgb, alpha, truth=None):
    
    rgb_image = np.full((height * width, 3), bgcolor, dtype='float32')
    truth_image = np.full((height * width, 3), bgcolor, dtype='float32')

    rgb_image[ray_mask] = rgb
    rgb_image = to_8b_image(rgb_image.reshape((height, width, 3)))

    if truth is not None:
        truth_image[ray_mask] = truth
        truth_image = to_8b_image(truth_image.reshape((height, width, 3)))

    alpha_map = unpack_alpha_map(alpha, ray_mask, width, height)
    alpha_image  = to_8b3ch_image(alpha_map)

    return rgb_image, alpha_image, truth_image

def _freeview(
        data_type='freeview',
        folder_name=None,
        action=None):
    cfg.perturb = 0.

    test_loader = create_dataloader(data_type)
    num_batches = len(test_loader)
    writer = ImageWriter(
                output_dir=os.path.join(cfg.logdir, cfg.load_net),
                exp_name=folder_name)
    joints_data = None
    for batch in test_loader:
        joints_data = batch['3d_joints']
        break  # Only load once

    data_batch_dict = {}  # Dictionary to store batch data

    for batch_idx, batch in tqdm(enumerate(test_loader), total = num_batches):
        for k, v in batch.items():
            batch[k] = v[0]

        data = cpu_data_to_gpu(
                    batch,
                    exclude_keys=EXCLUDE_KEYS_TO_GPU)

        with torch.no_grad():
            net_output = model(**data, 
                               iter_val=cfg.eval_iter)
        rgb = net_output['rgb']

        alpha = net_output['alpha']

        width = batch['img_width']
        height = batch['img_height']
        ray_mask = batch['ray_mask']

        target_rgbs = batch.get('target_rgbs', None)

        rgb_img, alpha_img, _ = unpack_to_image(
            width, height, ray_mask, np.array(cfg.bgcolor) / 255.,
            rgb.data.cpu().numpy(),
            alpha.data.cpu().numpy())

        imgs = [rgb_img]
        if cfg.show_truth and target_rgbs is not None:
            target_rgbs = to_8b_image(target_rgbs.numpy())
            imgs.append(target_rgbs)
        if cfg.show_alpha:
            imgs.append(alpha_img)

        img_out = np.concatenate(imgs, axis=1)
        writer.append(img_out)

        key = f"{batch_idx:06d}"
        batch_data = {
                'extrinsic': batch['extrinsic'].cpu().numpy(),
                'intrinsic': batch['intrinsic'].cpu().numpy(),
                'joints': joints_data.cpu().numpy()
            }
        data_batch_dict[key] = batch_data
    pickle_file_path = os.path.join(cfg.logdir, cfg.load_net, folder_name, 'extrinsic_intrinsic_joints.pkl')
    with open(pickle_file_path, 'wb') as pickle_file:
        pickle.dump(data_batch_dict, pickle_file)
    print("Dumping is Finished")

    writer.finalize()

h36m_actions = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo', 'Posing', 'Purchases',
                'Sitting', 'SittingDown', 'Smoking', 'Waiting', 'WalkDog', 'Walking', 'WalkTogether']

if __name__ == '__main__':
    model = load_network()
    model.eval()

    imgs_dir = Path('dataset', cfg.task, cfg.subject, cfg.action, 'images')
    num_frames = len(os.listdir(imgs_dir))
    for i in range(num_frames):
        cfg.freeview.frame_idx = i
        _freeview(data_type='freeview',
                  folder_name=f"{cfg.action}/topview_{5*cfg.freeview.frame_idx:06d}/H{cfg.topview_camera.h}R{cfg.topview_camera.r}"
                 )
