import os
import pickle
import random
import numpy as np
import cv2
import torch
import torch.utils.data

from core.utils.image_util import load_image
from core.utils.body_util import \
    body_pose_to_body_RTs, \
    get_canonical_global_tfms, \
    approx_gaussian_bone_volumes
from core.utils.camera_util import \
    rotate_camera_by_frame_idx, \
    apply_global_tfm_to_camera, \
    get_rays_from_KRT,\
    get_rays_from_KRT_adapted_from_original, \
    rays_intersect_3d_bbox
from core.utils.file_util import list_files, split_path

from configs import cfg
	 			
h36m_matrix = np.array([[1., 0., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., -1., 0.6],
                        [0.,0.,0.,1.]])

genebody_matrix = np.array([[1., 0., 0., 0.],
                            [0., 0., -1., 1.],
                            [0., 1., 0., 0.6],
                            [0.,0.,0.,1.]])

zju_matrix = np.array([[-1., 0., 0., 0.],
                        [0., -1., 0., 0.],
                        [0., 0., 1., 3.1],
                        [0.,0.,0.,1.]])

zju_matrix_390 = np.array([[-1., 0., 0., 0.],
                        [0., -1., 0., 0.],
                        [0., 0., 1., 3.],
                        [0.,0.,0.,1.]])

zju_matrix_rev = np.array([[-1., 0., 0., 0.],
                        [0., -1., 0., 0.],
                        [0., 0., -1., 1.],
                        [0.,0.,0.,1.]])

def move_camera_by_frame_idx(extrinsics, 
                             frame_idx, 
                             r,
                             rotate_axis='z',
                             trans=None,
                             period=196,
                             inv_angle=False):
    r""" Get camera extrinsics based on frame index and rotation period.

    Args:
        - extrinsics: Array (3, 3)
        - frame_idx: Integer
        - trans: Array (3, )
        - rotate_axis: String
        - period: Integer
        - inv_angle: Boolean (clockwise/counterclockwise)

    Returns:
        - Array (3, 3)
    """
    ext_new = extrinsics.copy()
    if trans is not None:
        ext_new[:3, 3] += trans.T
    
    if frame_idx == 0:
        return ext_new
    else:
        angle = 2 * np.pi * (frame_idx - 1) / (period - 1)
        ext_new[0, 3] += r * np.cos(angle)
        ext_new[1, 3] += r * np.sin(angle)
        return ext_new
        
class Dataset(torch.utils.data.Dataset):
    ROT_CAM_PARAMS = {
        'zju_mocap': {'rotate_axis': 'z','inv_angle': False},
        'wild': {'rotate_axis': 'y', 'inv_angle': False},
        'h36m': {'rotate_axis': 'z','inv_angle': True},
        'genebody': {'rotate_axis': 'y','inv_angle': True}
    }

    def __init__(
            self, 
            dataset_path,
            keyfilter=None,
            maxframes=-1,
            skip=1,
            bgcolor=None,
            src_type="zju_mocap",
            **_):

        print('[Dataset Path]', dataset_path) 

        self.dataset_path = dataset_path
        self.image_dir = os.path.join(dataset_path, 'images')

        self.canonical_joints, self.canonical_bbox = \
            self.load_canonical_joints()

        if 'motion_weights_priors' in keyfilter:
            self.motion_weights_priors = \
                approx_gaussian_bone_volumes(
                    self.canonical_joints, 
                    self.canonical_bbox['min_xyz'],
                    self.canonical_bbox['max_xyz'],
                    grid_size=cfg.mweight_volume.volume_size).astype('float32')

        if src_type == 'h36m':
            cameras = self.new_train_cameras(h36m_matrix)
        elif src_type == 'genebody':
            cameras = self.new_train_cameras(genebody_matrix)
        elif src_type == 'zju_mocap':
            cameras = self.new_train_cameras(zju_matrix)
            if dataset_path.lstrip('dataset/zju_mocap/') in ['313', '315']:
                cameras = self.new_train_cameras(zju_matrix_rev)
            elif dataset_path.lstrip('dataset/zju_mocap/') in ['390']:
                cameras = self.new_train_cameras(zju_matrix_390)
        else:
            print("Incorrect dataset")
        #cameras = self.load_train_cameras()
        mesh_infos = self.load_train_mesh_infos()

        framelist = self.load_train_frames() 
        self.framelist = framelist[::skip]
        if maxframes > 0:
            self.framelist = self.framelist[:maxframes]  

        self.train_frame_idx = cfg.freeview.frame_idx
        print(f' -- Frame Idx: {self.train_frame_idx}')

        self.total_frames = cfg.render_frames
        print(f' -- Total Rendered Frames: {self.total_frames}')

        self.train_frame_name = framelist[self.train_frame_idx]
        
        self.train_camera = cameras[framelist[self.train_frame_idx]]
        
        self.train_mesh_info = mesh_infos[framelist[self.train_frame_idx]]

        self.bgcolor = bgcolor if bgcolor is not None else [255., 255., 255.]
        self.keyfilter = keyfilter
        self.src_type = src_type

    def load_canonical_joints(self):
        cl_joint_path = os.path.join(self.dataset_path, 'canonical_joints.pkl')
        with open(cl_joint_path, 'rb') as f:
            cl_joint_data = pickle.load(f)
        canonical_joints = cl_joint_data['joints'].astype('float32')
        canonical_bbox = self.skeleton_to_bbox(canonical_joints)

        return canonical_joints, canonical_bbox


    def new_train_cameras(self, matrix):
#        print(matrix)
        
        new_camera = self.load_train_cameras()
        for keys in new_camera:
            new_camera[keys]['extrinsics'] = matrix
        return new_camera

    def load_train_cameras(self):
        cameras = None
        with open(os.path.join(self.dataset_path, 'cameras.pkl'), 'rb') as f: 
            cameras = pickle.load(f)   
        return cameras

    @staticmethod
    def skeleton_to_bbox(skeleton):
        min_xyz = np.min(skeleton, axis=0) - cfg.bbox_offset
        max_xyz = np.max(skeleton, axis=0) + cfg.bbox_offset

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }

    def load_train_mesh_infos(self):
        mesh_infos = None
        with open(os.path.join(self.dataset_path, 'mesh_infos.pkl'), 'rb') as f:   
            mesh_infos = pickle.load(f)

        for frame_name in mesh_infos.keys():
            bbox = self.skeleton_to_bbox(mesh_infos[frame_name]['joints'])
            mesh_infos[frame_name]['bbox'] = bbox

        return mesh_infos

    def load_train_frames(self):
        img_paths = list_files(os.path.join(self.dataset_path, 'images'),
                               exts=['.png'])
        return [split_path(ipath)[1] for ipath in img_paths]
    
    def query_dst_skeleton(self):
        return {
            'poses': self.train_mesh_info['poses'].astype('float32'),
            'dst_tpose_joints': \
                self.train_mesh_info['tpose_joints'].astype('float32'),
            'bbox': self.train_mesh_info['bbox'].copy(),
            'Rh': self.train_mesh_info['Rh'].astype('float32'),
            'Th': self.train_mesh_info['Th'].astype('float32'),
            'dst_joints': self.train_mesh_info['joints'].astype('float32')
        }

    def get_freeview_camera(self, frame_idx, total_frames, trans=None):
        E = rotate_camera_by_frame_idx(
                extrinsics=self.train_camera['extrinsics'], 
                frame_idx=frame_idx,
                period=total_frames,
                trans=trans,
                **self.ROT_CAM_PARAMS[self.src_type])

        K = self.train_camera['intrinsics'].copy()
        D = self.train_camera['distortions'].copy()
        K[:2] *= cfg.resize_img_scale
        return K, E, D
    
    def get_circleview_camera(self, frame_idx, total_frames, r=1, trans=None):
        E = move_camera_by_frame_idx(extrinsics=self.train_camera['extrinsics'], 
                                     frame_idx=frame_idx,
                                     period=total_frames,
                                     r=r,
                                     trans=trans,
                                     **self.ROT_CAM_PARAMS[self.src_type])

        K = self.train_camera['intrinsics'].copy()
        D = self.train_camera['distortions'].copy()
        K[:2] *= cfg.resize_img_scale
        return K, E, D

    def load_image(self, frame_name, bg_color):
        imagepath = os.path.join(self.image_dir, '{}.png'.format(frame_name))
        orig_img = np.array(load_image(imagepath))

        maskpath = os.path.join(self.dataset_path, 
                                'masks', 
                                '{}.png'.format(frame_name))
        alpha_mask = np.array(load_image(maskpath))
        
        if 'distortions' in self.train_camera:
            K = self.train_camera['intrinsics']
            D = self.train_camera['distortions']
            orig_img = cv2.undistort(orig_img, K, D)
            alpha_mask = cv2.undistort(alpha_mask, K, D)

        alpha_mask = alpha_mask / 255.
        img = alpha_mask * orig_img + (1.0 - alpha_mask) * bg_color[None, None, :]
        if cfg.resize_img_scale != 1.:
            img = cv2.resize(img, None, 
                             fx=cfg.resize_img_scale,
                             fy=cfg.resize_img_scale,
                             interpolation=cv2.INTER_LANCZOS4)
            alpha_mask = cv2.resize(alpha_mask, None, 
                                    fx=cfg.resize_img_scale,
                                    fy=cfg.resize_img_scale,
                                    interpolation=cv2.INTER_LINEAR)
                                
        return img, alpha_mask

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        frame_name = self.train_frame_name
        results = {
            'frame_name': frame_name
        }

        bgcolor = np.array(self.bgcolor, dtype='float32')

        img, _ = self.load_image(frame_name, bgcolor)
        img = img / 255.
        H, W = img.shape[0:2]

        dst_skel_info = self.query_dst_skeleton()
        dst_bbox = dst_skel_info['bbox']
        dst_poses = dst_skel_info['poses']

        dst_tpose_joints = dst_skel_info['dst_tpose_joints']
        dst_Rh = dst_skel_info['Rh']
        dst_Th = dst_skel_info['Th']
        dst_joints = dst_skel_info['dst_joints']
        #print(dst_Th)

        dst_Th = np.array([0., 0., 1.])  # This is the line that you need to add
        
        camera_r = cfg.topview_camera.r if 'topview_camera' in cfg.keys() else 1
        K, E, D = self.get_circleview_camera(
                        frame_idx=idx,
                        total_frames=self.total_frames,
                        trans=dst_Th,
                        r=camera_r)
        '''
        K[0,0] = 2 * H/np.pi
        K[1,1] = 2 * W/np.pi
        '''
        Cx = K[0, 2]
        Cy = K[1, 2]
        #focal = 2.*H/np.pi
        focal = H / np.pi
        K = np.array([[focal, 0, Cx],
                      [0, focal, Cy],
                      [0, 0, 1]])
        
        
        E = apply_global_tfm_to_camera(
                E=E, 
                Rh=dst_Rh,
                Th=dst_Th)

        R = E[:3, :3]
        T = E[:3, 3]
        if 'topview_camera' in cfg.keys():
            T[2] = cfg.topview_camera.h

        rays_o, rays_d = get_rays_from_KRT_adapted_from_original(H, W, K, R, T)
#        rays_o, rays_d = get_rays_from_KRT(H, W, K, R, T)
        rays_o = rays_o.reshape(-1, 3) # (H, W, 3) --> (N_rays, 3)
        rays_d = rays_d.reshape(-1, 3)

        # (selected N_samples, ), (selected N_samples, ), (N_samples, )
        near, far, ray_mask = rays_intersect_3d_bbox(dst_bbox, rays_o, rays_d)
        rays_o = rays_o[ray_mask]
        rays_d = rays_d[ray_mask]

        near = near[:, None].astype('float32')
        far = far[:, None].astype('float32')
    
        batch_rays = np.stack([rays_o, rays_d], axis=0) 

        if 'rays' in self.keyfilter:
            results.update({
                'img_width': W,
                'img_height': H,
                'extrinsic': E,
                'intrinsic': K,
                '3d_joints': dst_joints,
                'ray_mask': ray_mask,
                'rays': batch_rays,
                'near': near,
                'far': far,
                'bgcolor': bgcolor})

        if 'target_rgbs' in self.keyfilter:
            results['target_rgbs'] = img

        if 'motion_bases' in self.keyfilter:
            dst_Rs, dst_Ts = body_pose_to_body_RTs(
                    dst_poses, dst_tpose_joints)
            cnl_gtfms = get_canonical_global_tfms(self.canonical_joints)
            results.update({
                'dst_Rs': dst_Rs,
                'dst_Ts': dst_Ts,
                'cnl_gtfms': cnl_gtfms
            })                                    

        if 'motion_weights_priors' in self.keyfilter:
            results['motion_weights_priors'] = \
                self.motion_weights_priors.copy()

        if 'cnl_bbox' in self.keyfilter:
            min_xyz = self.canonical_bbox['min_xyz'].astype('float32')
            max_xyz = self.canonical_bbox['max_xyz'].astype('float32')
            results.update({
                'cnl_bbox_min_xyz': min_xyz,
                'cnl_bbox_max_xyz': max_xyz,
                'cnl_bbox_scale_xyz': 2.0 / (max_xyz - min_xyz)
            })
            assert np.all(results['cnl_bbox_scale_xyz'] >= 0)

        if 'dst_posevec_69' in self.keyfilter:
            # 1. ignore global orientation
            # 2. add a small value to avoid all zeros
            dst_posevec_69 = dst_poses[3:] + 1e-2
            results.update({
                'dst_posevec': dst_posevec_69,
            })


        return results
