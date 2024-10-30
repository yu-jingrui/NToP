import pickle
import json
import os
import glob
import shutil
from pathlib import Path

import cv2
import torch
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import binary_opening, binary_erosion, binary_dilation

def gen_xyxybbox_from_alpha(alpha_f):
    aimg = Image.open(alpha_f)
    binimg = aimg.convert("1")
    mask = np.nonzero(binimg)
    y1, y2, x1, x2 = (float(min(mask[0])), float(max(mask[0])), float(min(mask[1])), float(max(mask[1])))
    
    return [x1, y1, x2, y2]

def gen_xyxybbox_area_from_alpha(alpha_f, slack=None):
    aimg = Image.open(alpha_f)
    binimg = aimg.convert("1")
    mask = np.nonzero(binimg)
    y1, y2, x1, x2 = (float(min(mask[0])), float(max(mask[0])), float(min(mask[1])), float(max(mask[1])))
    
    if slack is None:
        slack = 0.1
        
    xslack = (x2 - x1) * slack * 0.5
    yslack = (y2 - y1) * slack * 0.5
    
    x1 = max(x1 - xslack, 0)
    y1 = max(y1 - yslack, 0)
    x2 = min(x2 + xslack, aimg.size[0])
    y2 = min(y2 + yslack, aimg.size[1])
    
    return [x1, y1, x2, y2], float(np.sum(binimg))

def gen_xyxybbox_from_rgb(rgb_f):
    rgbimg = Image.open(rgb_f)
    greyimg = rgbimg.convert('1')
    greyimg = np.invert(greyimg)
    greyimg = binary_dilation(greyimg, structure=np.ones((2,2)))
    greyimg = binary_erosion(greyimg, structure=np.ones((3,3)))
    mask = np.argwhere(np.not_equal(greyimg, False))
    y1, x1, y2, x2 = (float(min(mask[:,0])), float(min(mask[:,1])), float(max(mask[:,0])), float(max(mask[:,1])))
    
    return [x1, y1, x2, y2]

def gen_xyxybbox_area_from_rgb(rgb_f, slack=None):
    rgbimg = Image.open(rgb_f)
    greyimg = rgbimg.convert('1')
    greyimg = np.invert(greyimg)
    greyimg = binary_dilation(greyimg, structure=np.ones((2,2)))
    greyimg = binary_erosion(greyimg, structure=np.ones((3,3)))
    mask = np.argwhere(np.not_equal(greyimg, False))
    y1, x1, y2, x2 = (float(min(mask[:,0])), float(min(mask[:,1])), float(max(mask[:,0])), float(max(mask[:,1])))
        
    if slack is None:
        slack = 0.1
        
    xslack = (x2 - x1) * slack * 0.5
    yslack = (y2 - y1) * slack * 0.5
    
    x1 = max(x1 - xslack, 0)
    y1 = max(y1 - yslack, 0)
    x2 = min(x2 + xslack, rgbimg.size[0])
    y2 = min(y2 + yslack, rgbimg.size[1])
    
    return [x1, y1, x2, y2], float(np.sum(greyimg))


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

def plot_bbox(xyxybbox, ax):
    x1, y1, x2, y2 = xyxybbox
    w = x2 - x1
    h = y2 - y1
    rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    return ax

def vis_skeleton_single_image(keypoints, image_path=None, bbox=None):
    kpts2d = np.array(keypoints)

    _, ax = plt.subplots(1, 1)
    if image_path is not None:
        img = cv2.imread(image_path)
        ax.imshow(img[..., ::-1])
        H, W = img.shape[:2]
    else:
        H, W = 300, 300
#    plotSkel2D(kpts2d, ax = ax)
    plotSkel2D(kpts2d, skeleton_tree, ax, linewidth=1, alpha=1, max_range=1, thres=0.5 )
    if bbox is not None:
        plot_bbox(bbox, ax)
    plt.show()
    
def skel3dplot(kps, config, ax = None, phi = 0, theta = 0, linewidth = 4, color = None, max_range = 1):
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

    
def create_cocolike_pose_anno(ntop_src, subset, subj, action=None, posetype='coco'):
    subsets_idx = {'human36m': 1, 'genebody': 2, 'zjumocap': 3}
    h36m_actions_idx = {'Directions': '001', 'Discussion': '002', 'Eating': '003',
                         'Greeting': '004', 'Phoning': '005', 'Photo': '006',
                         'Posing': '007', 'Purchases': '008', 'Sitting': '009',
                         'SittingDown': '010', 'Smoking': '011', 'Waiting': '012',
                         'WalkDog': '013', 'WalkTogether': '014', 'Walking': '015'}
    genebody_subjects_idx = {'ahha': '001', 'alejandro': '002', 'anastasia': '003',
                              'aosilan': '004', 'arslan': '005', 'barlas': '006',
                              'barry': '007', 'camilo': '008', 'dannier': '009',
                              'gaoxing': '010', 'huajiangtao5': '011', 'joseph': '012',
                              'kamal_ejaz': '013', 'kemal': '014', 'lihongyun': '015',
                              'natacha': '016', 'quyuanning': '017', 'rabbi': '018',
                              'rivera': '019', 'songyujie': '020', 'sunyuxing': '021',
                              'wuwenyan': '022', 'xujiarui': '023', 'zhanghao': '024',
                              'zhanghongwei': '025', 'zhangziyu': '026', 'zhuna2': '027'}
    
    dataset_idx = subsets_idx[subset]
    
    if subset == 'human36m':
        subj_idx = f'{int(subj.lstrip("S")):03d}'
    elif subset == 'zjumocap':
        subj_idx = subj.lstrip("p")
    else:
        subj_idx = genebody_subjects_idx[subj]
    
    
    if subset == 'human36m' and action != None:
        ac_idx = h36m_actions_idx[action]
    else:
        ac_idx = '000'
    
    kp_annos = {
    "info": {"description": "NTOP",
             "url": "https://www.tu-chemnitz.de/etit/dst/forschung/comp_vision/datasets/NTOP/",
             "version": "1.0",
             "year": "2023", "contributor": "Chair of Digital und Circuit Design",
             "date_created": "11/10/23"},
    "licenses": [{"url": "https://creativecommons.org/licenses/by/4.0/",
                  "id": 1,
                  "name": "Attribution 4.0 International (CC BY 4.0)"
                 }],
    }
    if posetype == 'coco':
        kp_annos["categories"] = [{"supercategory": "person",
                                    "id": 1,
                                    "name": "person",
                                    "keypoints": ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                                                  "left_shoulder", "right_shoulder",
                                                  "left_elbow", "right_elbow",
                                                  "left_wrist", "right_wrist",
                                                  "left_hip", "right_hip",
                                                  "left_knee", "right_knee",
                                                  "left_ankle", "right_ankle"],
                                    "skeleton": [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                                                 [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10],
                                                 [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
                                   }]
    elif posetype == 'smpl':
        kp_annos["categories"] = [{"supercategory": "person",
                                    "id": 1,
                                    "name": "person",
                                    "keypoints": ['MidHip', 'LUpLeg', 'RUpLeg', 'spine',
                                                  'LLeg', 'RLeg', 'spine1', 'LFoot','RFoot',
                                                  'spine2', 'LToeBase', 'RToeBase', 'neck',
                                                  'LShoulder', 'RShoulder', 'head',
                                                  'LArm', 'RArm', 'LForeArm', 'RForeArm',
                                                  'LHand', 'RHand', 'LHandIndex1', 'RHandIndex1', ],
                                    "skeleton": [[ 0, 1 ],[ 0, 2 ],[ 0, 3 ],[ 3, 6 ],[ 6, 9 ],
                                                 [ 1, 4 ],[ 4, 7 ],[ 7, 10],[ 2, 5 ],[ 5, 8 ],
                                                 [ 8, 11],[ 9, 12],[12, 15],[ 9, 13],[13, 16],
                                                 [16, 18],[18, 20],[20, 22],[ 9, 14],[14, 17],
                                                 [17, 19],[19, 21],[21, 23]]
                                   }]
    else:
        print(f'Unsupported pose type: {posetype}. Annotation generation exited!')
        return
    
    img_dir = Path(ntop_src, subset, 'concat', subj, 'rgb')
    img_files = sorted(os.listdir(img_dir))
    if ac_idx != '000':
        img_files = [f for f in img_files if f[6:9] == ac_idx]
    with Image.open(Path(img_dir, img_files[0])) as im:
        w, h = im.size

    images = []
    for img in img_files:
        img_dict = {
            'file_name': img,
            'width': w,
            'height': h,
            'id': int(img.rstrip('.png'))
        }
        images.append(img_dict)
    kp_annos['images'] = images
    
    def _collect_cocolike_anno_list(anno_file):
        annotations = []
        bbox_annos = []
        with open(anno_file) as f:
            annos = json.load(f)
        renders = list(annos.keys())
        for rd in renders:
            for hr in annos[rd]:
                if len(hr) == 8:
                    hr_str = hr[1] + hr[3] + hr[5] + hr[7]
                else:
                    hr_str = hr[1] + hr[3] + hr[5] + '0'
                anno = annos[rd][hr]
                for num in range(9):
                    image_id_str = f'{dataset_idx:03d}{subj_idx}{ac_idx}{rd.lstrip("topview_")}{hr_str}{num}'
                    if subset == 'human36m':
                        rgb_f = Path(img_dir, image_id_str+'.png')
                        bbox, area = gen_xyxybbox_area_from_rgb(rgb_f)
                    else:
                        alpha_f = Path(ntop_src, subset, 'concat', subj, 'alpha', image_id_str+'.png')
                        bbox, area = gen_xyxybbox_area_from_alpha(alpha_f, slack=0.05)
                    kps = np.array(anno[num][posetype+'_2d'])
                    kps_flat = kps.flatten()
                    if posetype == 'coco':
                        kps_flat[3:15] = 0.0
                    kps_flat = list(kps_flat)
                    annotation = {
                        'id': int(image_id_str),
                        'image_id': int(image_id_str),
                        'category_id': 1,
                        'segmentation': [],
                        'area': area,
                        'bbox': bbox,
                        'iscrowd': 0,
                        'num_keypoints': 13 if posetype=='coco' else 24,
                        'keypoints': kps_flat
                    }
                    bb_anno = {"category_id": 1, "image_id": int(image_id_str), "score": 1, "bbox": bbox}
                    annotations.append(annotation)
                    bbox_annos.append(bb_anno)
        return annotations, bbox_annos
    
    if ac_idx != '000':
        anno_file = Path(ntop_src, subset, 'anno', f'{subj}_{action}_anno.json')
        annotations, bbox_annos = _collect_cocolike_anno_list(anno_file)
    elif subset == 'human36m':
        annotations = []
        bbox_annos = []
        for ac, ac_idx in h36m_actions_idx.items():
            anno_file = Path(ntop_src, subset, 'anno', f'{subj}_{ac}_anno.json')
            ans, bans = _collect_cocolike_anno_list(anno_file)
            annotations += ans
            bbox_annos += bans
    else:
        anno_file = Path(ntop_src, subset, 'anno', f'{subj}_anno.json')
        annotations, bbox_annos = _collect_cocolike_anno_list(anno_file)
        
    kp_annos['annotations'] = annotations
    
    return kp_annos, bbox_annos

def create_hybrik_pose_anno(ntop_src, subset, subj):
    subsets_idx = {'human36m': 1, 'genebody': 2, 'zjumocap': 3}
    h36m_actions_idx = {'Directions': '001', 'Discussion': '002', 'Eating': '003',
                         'Greeting': '004', 'Phoning': '005', 'Photo': '006',
                         'Posing': '007', 'Purchases': '008', 'Sitting': '009',
                         'SittingDown': '010', 'Smoking': '011', 'Waiting': '012',
                         'WalkDog': '013', 'WalkTogether': '014', 'Walking': '015'}
    genebody_subjects_idx = {'ahha': '001', 'alejandro': '002', 'anastasia': '003',
                              'aosilan': '004', 'arslan': '005', 'barlas': '006',
                              'barry': '007', 'camilo': '008', 'dannier': '009',
                              'gaoxing': '010', 'huajiangtao5': '011', 'joseph': '012',
                              'kamal_ejaz': '013', 'kemal': '014', 'lihongyun': '015',
                              'natacha': '016', 'quyuanning': '017', 'rabbi': '018',
                              'rivera': '019', 'songyujie': '020', 'sunyuxing': '021',
                              'wuwenyan': '022', 'xujiarui': '023', 'zhanghao': '024',
                              'zhanghongwei': '025', 'zhangziyu': '026', 'zhuna2': '027'}
    
    dataset_idx = subsets_idx[subset]
    
    if subset == 'human36m':
        subj_idx = f'{int(subj.lstrip("S")):03d}'
    elif subset == 'zjumocap':
        subj_idx = subj.lstrip("p")
    else:
        subj_idx = genebody_subjects_idx[subj]
        
    hybrik_annos = {
        'categories': [{'H36M_TO_J14': [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10],
                        'H36M_TO_J17': [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9],
                        'J24_FLIP_PERM': [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6,
                                          12, 13, 14, 15, 16, 17, 18, 19, 21, 20, 23, 22],
                        'J24_TO_J14': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18],
                        'J24_TO_J17': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17],
                        'SMPL_JOINTS_FLIP_PERM': [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12,
                                                  14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22],
                        'id': 1,
                        'name': 'person',
                        'supercategory': 'person'}]
        }
    
    if subset == 'human36m':
        raw_annos_everything = {}
        for ac, ac_idx in h36m_actions_idx.items():
            raw_anno_f = Path(ntop_src, subset, 'anno', f'{subj}_{ac}_anno.json')
            with open(raw_anno_f, 'r') as f:
                raw_anno = json.load(f)
            renders = list(raw_anno.keys())
            for rd in renders:
                for hr in raw_anno[rd]:
                    hr_str = hr[1] + hr[3] + hr[5] + '0'
                    anno = raw_anno[rd][hr]
                    for num in range(9):
                        image_id_str = f'{dataset_idx:03d}{subj_idx}{ac_idx}{rd.lstrip("topview_")}{hr_str}{num}'
                        ranno = anno[num].copy()
                        ranno.update({'action_name':ac, 'ac_idx':int(ac_idx)})
                        raw_annos_everything[image_id_str] = ranno
    else:
        raw_annos_everything = {}
        ac_idx = '000'
        raw_anno_f = Path(ntop_src, subset, 'anno', f'{subj}_anno.json')
        with open(raw_anno_f, 'r') as f:
            raw_anno = json.load(f)
        renders = list(raw_anno.keys())
        for rd in renders:
            for hr in raw_anno[rd]:
                hr_str = hr[1] + hr[3] + hr[5] + hr[7]
                anno = raw_anno[rd][hr]
                for num in range(9):
                    image_id_str = f'{dataset_idx:03d}{subj_idx}{ac_idx}{rd.lstrip("topview_")}{hr_str}{num}'
                    ranno = anno[num].copy()
                    ranno.update({'action_name':None, 'ac_idx':int(ac_idx)})
                    raw_annos_everything[image_id_str] = ranno
        
    coco_anno_f = Path(ntop_src, subset, 'anno', f'{subj}_anno_coco.json')
    with open(coco_anno_f, 'r') as f:
        coco_anno = json.load(f)
    annotations = coco_anno['annotations'].copy()
    images = coco_anno['images'].copy()
    
    for img, an in zip(images, annotations):
        img_id_str = f"00{img['id']}"
        if img_id_str not in raw_annos_everything.keys():
            print(f'{image_id_str} not found in raw annos')
            pass
        raw = raw_annos_everything[img_id_str]

        an['thetas'] = raw['thetas']
        an['betas'] = raw['betas']
        an['root_coord'] = raw['root_coord']
        smpl_kps = raw['joints_3d']
        smpl_kps = [j[0:3] for j in smpl_kps]
        smpl_kps_flat = list(np.array(smpl_kps).flatten())
        an['smpl_keypoints'] = smpl_kps_flat
        del an['keypoints']
        del an['num_keypoints']

        img['cam_idx'] = int(str(img['id'])[-1])
        img['cam_param'] = raw['cam_param']
        img['subject'] = subj
        img['subject_idx'] = int(str(dataset_idx)+subj_idx)
        img['action_name'] = raw['action_name']
        img['action_idx'] = raw['ac_idx']
        
    hybrik_annos['images'] = images
    hybrik_annos['annotations'] = annotations
    
    return hybrik_annos