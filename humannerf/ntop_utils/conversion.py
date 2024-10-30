import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
#import os
#from mpl_toolkits.mplot3d import Axes3D
#import open3d as o3d
#from sklearn.manifold import TSNE
#import matplotlib.patches as patches
from mmhuman3d.core.conventions.keypoints_mapping import KEYPOINTS_FACTORY, convert_kps
#import pickle
#import io
#import plotly.graph_objects as go
#from PIL import Image
#from collections import deque
#from copy import copy, deepcopy
#from collections import namedtuple
#import torch.nn.functional as F
#import pytorch3d.transforms.rotation_conversions as p3drs
from scipy.spatial.transform import Rotation

## Skeleton-connection

skeleton_tree = {
    'color': [
        'g', 'r', 'r', 'r', 'r', 'b', 'b', 'b','b', 'r', 'r', 'b', 'b'
    ],
    'coco_tree': [[0, 1], [4,5], [4,11], [5,7], [7,9], [4,6], [4,12], [6,8], [8,10], [11,13], [13,15], [12,14], [14,16]],
    
    'smpl_color': [
        'g', 'g', 'g', 'g', 'g', 'r', 'r', 'r', 'b', 'b', 'b', 'g', 'g', 'r',
        'r', 'r', 'r', 'r', 'b', 'b', 'b', 'b', 'b', 'b'
    ],
    'smpl_tree': [[ 0, 1 ],[ 0, 2 ],[ 0, 3 ],[ 3, 6 ],[ 6, 9 ],[ 1, 4 ],[ 4, 7 ],[ 7, 10],[ 2, 5 ],[ 5, 8 ],
        [ 8, 11],[ 9, 12],[12, 15],[ 9, 13],[13, 16],[16, 18],[18, 20],[20, 22],[ 9, 14],[14, 17],[17, 19],[19, 21],[21, 23]]
}

## Interpolated_nose_joint

def nose_predict(joints, z, y, x):
    
    # Indices of the neck/head joint and other relevant joints
    neck_head_idx = 15  # Adjust this index based on SMPL joint structure
    
    # Get neck/head joint position
    neck_head_pos = joints[0][neck_head_idx]
    
    # Interpolate or shift to get the nose position
    t = 0.2  # Interpolation parameter (0.0 to 1.0)
    interpolated_nose_pos = neck_head_pos * (1 + t)  # Assuming you want to shift along the line connecting neck/head and nose
    if not isinstance(z, float):
        raise ValueError("z must be a float")
    if not isinstance(y, float):
        raise ValueError("y must be a float")
    if not isinstance(x, float):
        raise ValueError("x must be a float")
    
    yaw_angle = z  # Adjust these angles as needed
    pitch_angle = y
    roll_angle = x
    
    # Create a rotation matrix based on the specified angles
    rotation = Rotation.from_euler('zyx', [yaw_angle, pitch_angle, roll_angle], degrees=False)
    
    # Apply the rotation to the neck/head position to get the aligned nose position
    aligned_nose_pos = rotation.apply(interpolated_nose_pos)

    joints[0][neck_head_idx] = aligned_nose_pos
    return joints


def coco_convert(joints):
    joints_copy = joints.copy()
    joints_copy= nose_predict(joints_copy, 0.0, 0.0, 0.07)
    coco_joints, mask = convert_kps(joints_copy, src='smpl', dst='coco')
    coco_joints[:, 0] = joints_copy[:, 15]
#    coco_joints[:, 11] = joints_copy[:, 1]
#    coco_joints[:, 12] = joints_copy[:, 2]
    return coco_joints

## 3d_to_2d_conversion
def nerf_c2w_to_extrinsic(c2w):
    return np.linalg.inv(swap_mat(c2w))

def swap_mat(mat):
    # [right, -up, -forward]
    # equivalent to right multiply by:
    # [1, 0, 0, 0]
    # [0,-1, 0, 0]
    # [0, 0,-1, 0]
    # [0, 0, 0, 1]
    return np.concatenate([
        mat[..., 0:1], -mat[..., 1:2], -mat[..., 2:3], mat[..., 3:]
        ], axis=-1)

def skeleton3d_to_2d(kp, c2w, H, W, focal, center=None):
    if kp.shape[1] == 17:
        kp =kp.reshape(17,3)
    else:
        kp = kp.reshape(24,3)
#    ext = np.array([nerf_c2w_to_extrinsic(c2w)])
    ext = np.array([c2w])
    f = focal if isinstance(focal, float) else focal[0]
    h = H if isinstance(H, int) else H
    w = W if isinstance(W, int) else W
    center = center if center is not None else (w * 0.5, h * 0.5)
    kp2d = world_to_cam_fisheye(kp, ext[0], h, w, focal, center)
    return kp2d

## world_cam_conversion function

def world_to_cam(pts, extrinsic, H, W, focal, center=None):
    if center is None:
        offset_x = W * 0.5
        offset_y = H * 0.5
    else:
        offset_x, offset_y = center

    if pts.shape[-1] < 4:
        pts = coord_to_homogeneous(pts)

    intrinsic = focal_to_intrinsic_np(focal)

    cam_pts = pts @ extrinsic.T @ intrinsic.T
    cam_pts = cam_pts[..., :2] / cam_pts[..., 2:3]
    cam_pts[cam_pts == np.inf] = 0.
    cam_pts[..., 0] += offset_x
    cam_pts[..., 1] += offset_y
    return cam_pts

## world_fisheye_cam_conversion function

def world_to_cam_fisheye(pts, extrinsic, H, W, focal, center=None):
    """
    Convert world points to image pixels for a fisheye camera.
    
    Args:
        - pts: Array (N, 3) of world points
        - extrinsic: Array (3, 4) of extrinsic camera matrix [R | T]
        - H: Height of the image
        - W: Width of the image
        - focal: Focal length of the fisheye camera
        - center: Center of the fisheye distortion (default is image center)
        
    Returns:
        - pixel_coords: Array (N, 2) of image pixel homogeneous coordinates 
    """
    
    if center is None:
        center = np.array([W / 2, H / 2])

    R = extrinsic[:, :3]
    T = extrinsic[:, 3]
    f = focal[0]
    intrinsic = focal_to_intrinsic_np(focal)
    # Calculate the transformation from world to camera coordinates
    #pts_cam = np.dot(R, pts.T - T[:, np.newaxis])

    if pts.shape[-1] < 4:
        pts = coord_to_homogeneous(pts)
    pts_cam = pts @ extrinsic.T @ intrinsic.T
    
    pts_cam = pts_cam[..., :2] / pts_cam[..., 2:3]
    # Convert to polar coordinates
    pts_cam[:,:2]
    r = np.linalg.norm(pts_cam[:,:2], axis=1)
    #phi = np.arctan2(pts_cam[1], pts_cam[0])
    phi = np.arctan2(pts_cam[:,1], pts_cam[:,0])
    theta = np.arctan2(r, f)
    rho = theta * f
    
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    u = x + center[0]
    v = y + center[1]
    
    pixel_coords = np.stack([u, v], axis=-1)
    return pixel_coords

def coord_to_homogeneous(c):
    assert c.shape[-1] == 3

    if len(c.shape) == 2:
        h = np.ones((c.shape[0], 1)).astype(c.dtype)
        return np.concatenate([c, h], axis=1)
    elif len(c.shape) == 1:
        h = np.array([0, 0, 0, 1]).astype(c.dtype)
        h[:3] = c
        return h
    else:
        raise NotImplementedError(f"Input must be a 2-d or 1-d array, got {len(c.shape)}")

def focal_to_intrinsic_np(focal):
    if isinstance(focal, float):
        focal_x = focal_y = focal
    elif isinstance(focal, list) and len(focal) == 2:
        focal_x, focal_y = focal
    else:
        raise ValueError("focal must be a float or a list of two floats (focal_x, focal_y)")

    return np.array([[focal_x, 0, 0, 0],
                     [0, focal_y, 0, 0],
                     [0, 0, 1, 0]],
                    dtype=np.float32)

## 3d_plot function

def Skel3dplot(kps, config, ax = None, phi = 0, theta = 0, linewidth = 4, color = None, max_range = 1):
    if kps.shape[1] == 17:
        kps = kps.reshape(17,3)
    else:
        kps = kps.reshape(24,3)
    kps = kps[:, [0,2,1]]
    multi = False
    if torch.is_tensor(kps):
        if len(kps) == 3:
            print(">>> View Multiperson")
            multi = True
            if kps.shape[1] != 3:
                kps = kps.transpose(1,2)
        elif len(kps) == 2:
            if kps.shape[0] != 3:
                kps = kps.transpose(0,1)
        else:
            raise RuntimeError('Wrong shapes for Kps')
    else:
        if kps.shape[0] != 3:
            kps = kps.T
    # kps: bn, 3, NumOfPoints or (3, N)

    if ax is None:
        print("Creating figure >>> ")
        fig = plt.figure(figsize =[10,10])
        ax = fig.add_subplot(111, projection = '3d')

    if kps.shape[1] == 17:
        for idx, (i,j) in enumerate(config['coco_tree']):
            if multi:
                for b in range(kps.shape[0]):
                    ax.plot([kps[b][0][i], kps[b][0][j]],
                            [kps[b][1][i], kps[b][1][j]],
                            [kps[b][2][i], kps[b][2][j]],
                            lw=linewidth,
                            color=config['color'][idx] if color is None else color,
                            alpha=1)
            else:
                ax.plot([kps[0][i], kps[0][j]], [kps[1][i], kps[1][j]],
                        [kps[2][i], kps[2][j]],
                        lw=linewidth,
                        color=config['color'][idx],
                        alpha=1)
    else:
        for idx, (i,j) in enumerate(config['smpl_tree']):
            if multi:
                for b in range(kps.shape[0]):
                    ax.plot([kps[b][0][i], kps[b][0][j]],
                            [kps[b][1][i], kps[b][1][j]],
                            [kps[b][2][i], kps[b][2][j]],
                            lw=linewidth,
                            color=config['smpl_color'][idx] if color is None else color,
                            alpha=1)
            else:
                ax.plot([kps[0][i], kps[0][j]], [kps[1][i], kps[1][j]],
                        [kps[2][i], kps[2][j]],
                        lw=linewidth,
                        color=config['smpl_color'][idx],
                        alpha=1)    
    
    if multi:
        for b in range(kps.shape[0]):
            ax.scatter(kps[b][0], kps[b][1], kps[b][2], color = 'r', alpha = 1,  marker='o')
            for joint_idx, (x, y, z) in enumerate(zip(kps[b][0], kps[b][1], kps[b][2])):
                ax.text(x, y, z, str(joint_idx), fontsize=8, color='black', va='center', ha='center')
    else:
        ax.scatter(kps[0], kps[1], kps[2], color = 'r', s = 30,  marker='o')
        for joint_idx, (x, y, z) in enumerate(zip(kps[0], kps[1], kps[2])):
            ax.text(x, y, z, str(joint_idx), fontsize=8, color='black', va='center', ha='center')

    ax.view_init(phi, theta)
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-1.2, 1)

    plt.xlabel('x')
    plt.ylabel('y')

    return ax

def plotSkel2D(pts,
               config=skeleton_tree,
               ax=None,
               linewidth=2,
               alpha=1,
               max_range=1,
               imgshape=None,
               thres=0.1):

    if len(pts.shape) == 2:
        pts = pts[None, :, :]  #(nP, nJ, 2/3)
    elif len(pts.shape) == 3:
        pass
    else:
        raise RuntimeError('The dimension of the points is wrong!')
    if torch.is_tensor(pts):
        pts = pts.detach().cpu().numpy()
    if pts.shape[2] == 3 or pts.shape[2] == 2:
        pts = pts.transpose((0, 2, 1))
    # pts : bn, 2/3, NumOfPoints or (2/3, N)
    if ax is None:
        fig = plt.figure(figsize=[5, 5])
        ax = fig.add_subplot(111)

    if pts.shape[2] == 17:  
        if 'color' in config.keys():
            colors = config['color']
        else:
            colors = ['b' for _ in range(len(config['coco_tree']))]
    else:
        if 'smpl_color' in config.keys():
            colors = config['smpl_color']
        else:
            colors = ['b' for _ in range(len(config['smpl_tree']))]
        

    def inrange(imgshape, pts):
        if pts[0] < 5 or \
           pts[0] > imgshape[1] - 5 or \
           pts[1] < 5 or \
           pts[1] > imgshape[0] - 5:
            return False
        else:
            return True

    for nP in range(pts.shape[0]):
        if pts.shape[2] == 17:
            for idx, (i, j) in enumerate(config['coco_tree']):
                if pts.shape[1] == 3:  # with confidence
                    if np.min(pts[nP][2][[i, j]]) < thres:
                        continue
                    lw = linewidth * 2 * np.min(pts[nP][2][[i, j]])
                else:
                    lw = linewidth
                if imgshape is not None:
                    if inrange(imgshape, pts[nP, :, i]) and \
                        inrange(imgshape, pts[nP, :, j]):
                        pass
                    else:
                        continue
                ax.plot([pts[nP][0][i], pts[nP][0][j]],
                        [pts[nP][1][i], pts[nP][1][j]],
                        lw=lw,
                        color=colors[idx],
                        alpha=1)
        elif pts.shape[2] == 24:
            for idx, (i, j) in enumerate(config['smpl_tree']):
                if pts.shape[1] == 3:  # with confidence
                    if np.min(pts[nP][2][[i, j]]) < thres:
                        continue
                    lw = linewidth * 2 * np.min(pts[nP][2][[i, j]])
                else:
                    lw = linewidth
                if imgshape is not None:
                    if inrange(imgshape, pts[nP, :, i]) and \
                        inrange(imgshape, pts[nP, :, j]):
                        pass
                    else:
                        continue
                ax.plot([pts[nP][0][i], pts[nP][0][j]],
                        [pts[nP][1][i], pts[nP][1][j]],
                        lw=lw,
                        color=colors[idx],
                        alpha=1)
        # if pts.shape[1] > 2:
        ax.scatter(pts[nP][0], pts[nP][1], c='r', s=20)
    if False:
        ax.axis('equal')
        plt.xlabel('x')
        plt.ylabel('y')
    else:
        ax.axis('off')
    return ax

def vis_skeleton_single_image(image_path, keypoints):
    img = cv2.imread(image_path)
    kpts2d = np.array(keypoints)

    _, ax = plt.subplots(1, 1)
    ax.imshow(img[..., ::-1])
    H, W = img.shape[:2]

    plotSkel2D(kpts2d, skeleton_tree, ax = ax,  linewidth = 2, alpha = 1, max_range = 1, thres = 0.5 )
    plt.show()