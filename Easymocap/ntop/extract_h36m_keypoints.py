import os
import os.path as osp
import pickle
import json
import numpy as np
from cdflib import CDF
import cv2
import shutil
import re
from metadata import H36M_Metadata

RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"

def _infer_box(pose3d, camera, rootIdx):
    root_joint = pose3d[rootIdx, :]
    tl_joint = root_joint.copy()
    tl_joint[0] -= 1000.0
    tl_joint[1] -= 900.0
    br_joint = root_joint.copy()
    br_joint[0] += 1000.0
    br_joint[1] += 1100.0
    tl_joint = np.reshape(tl_joint, (1, 3))
    br_joint = np.reshape(br_joint, (1, 3))

    tl2d = _weak_project(tl_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()

    br2d = _weak_project(br_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()
    return np.array([tl2d[0], tl2d[1], br2d[0], br2d[1]])


def _weak_project(pose3d, fx, fy, cx, cy):
    pose2d = pose3d[:, :2] / pose3d[:, 2:3]
    pose2d[:, 0] *= fx
    pose2d[:, 1] *= fy
    pose2d[:, 0] += cx
    pose2d[:, 1] += cy
    return pose2d


def main(h36m_root, output_folder):
    subject_list = [1, 5, 6, 7, 8, 9, 11]
    action_list = [x for x in range(2, 17)]
    subaction_list = [x for x in range(2, 3)]
    camera_list = [x for x in range(1, 5)]

    joint_idx = [14, 13, 25, 26, 27, 17, 18, 19, 0, 1, 2, 3, 6, 7, 8, 15, 15, 15, 15, 9, 9, 8, 5 ,5 ,4]

    json_folder = output_folder
    os.makedirs(json_folder, exist_ok=True)
    with open(f'{h36m_root}/H36M-Toolbox/camera_data.pkl', 'rb') as f:
        camera_data = pickle.load(f)

    cnt = 0
    for s in subject_list:
        subject_folder = osp.join(json_folder, f'S{s}')
        os.makedirs(subject_folder, exist_ok=True)
        for a in action_list:
            for sa in subaction_list:
                for c in camera_list:
                    camera = camera_data[(s, c)]
                    camera_dict = {}
                    camera_dict['R'] = camera[0]
                    camera_dict['T'] = camera[1]
                    camera_dict['fx'] = camera[2][0]
                    camera_dict['fy'] = camera[2][1]
                    camera_dict['cx'] = camera[3][0]
                    camera_dict['cy'] = camera[3][1]
                    camera_dict['k'] = camera[4]
                    camera_dict['p'] = camera[5]

                    subdir_format = 'S_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}'
                    subdir = subdir_format.format(s, a, sa, c)

                    metadata = H36M_Metadata(f'{h36m_root}/H36M-Toolbox/metadata.xml')
                    basename = metadata.get_base_filename('S{:d}'.format(s), '{:d}'.format(a), '{:d}'.format(sa), metadata.camera_ids[c-1])
                    action_name = basename.split(".")[0]
                    action_name = re.sub(r'\d+', '', action_name).strip()

                    camera_name = basename.split(".")[1]
                    #print(f"basename is {basename}, and action_name is {action_name}, camera_name is {camera_name}")
                    annotname = basename + '.cdf'
                    
                    subject = 'S' + str(s)
                    
                    annofile3d = osp.join(f'{h36m_root}/extracted', subject, 'Poses_D3_Positions_mono_universal', annotname)
                    annofile3d_camera = osp.join(f'{h36m_root}/extracted', subject, 'Poses_D3_Positions_mono', annotname)
                    annofile2d = osp.join(f'{h36m_root}/extracted', subject, 'Poses_D2_Positions', annotname)

                    with CDF(annofile3d) as data:
                        pose3d = np.array(data['Pose'])
                        pose3d = np.reshape(pose3d, (-1, 32, 3))

                    with CDF(annofile3d_camera) as data:
                        pose3d_camera = np.array(data['Pose'])
                        pose3d_camera = np.reshape(pose3d_camera, (-1, 32, 3))

                    with CDF(annofile2d) as data:
                        pose2d = np.array(data['Pose'])
                        pose2d = np.reshape(pose2d, (-1, 32, 2))
                        
                    nposes = min(pose3d.shape[0], pose2d.shape[0])
                    image_format = 's_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}_{:06d}.jpg'

                    confidence = np.ones((nposes, 32, 1))
                    pose3d_camera_confidence = np.append(pose3d_camera, confidence, axis=2)
                    pose2d_confidence = np.append(pose2d, confidence, axis=2)
                    
                    print(RED + "Processing Files " + RESET + GREEN 
                          + f"Subject:{subject} Action:{action_name} Camera:{camera_name}" + RESET)

                    for i in range(nposes):
                        datum = {}
                        metadatum = {}
                        imagename = image_format.format(s, a, sa, c, i+1)
                        imagepath = osp.join(subdir, imagename)


                        if not osp.isfile(osp.join(f'{h36m_root}/images', imagepath)):
                            print(osp.join('images', imagepath))
                            print(RED + "no file found"+ RESET)
                        
                        if osp.isfile(osp.join(f'{h36m_root}/images', imagepath)):
        
                            source_path = osp.join(f'{h36m_root}/images', imagepath)
                            #print( RED + "Processing File " + RESET + GREEN + "Subject :" f'{subject}'+ "  " + "Action: " f'{action_name}'+  "  " + "Camera :" f'{camera_name}' + RESET)
                            image = cv2.imread(source_path)
                            height, width, _ = image.shape

                            datum['image'] = imagepath
                            datum['joints_2d'] = pose2d[i, joint_idx, :]
                            pose2d_new = pose2d_confidence[i, joint_idx, :]

                            datum['joints_3d'] = pose3d[i, joint_idx, :]

                            datum['joints_3d_camera'] = pose3d_camera[i, joint_idx, :]

                            datum['joints_vis'] = np.ones((17, 3))
                            datum['video_id'] = cnt
                            datum['image_id'] = i+1
                            datum['subject'] = s
                            datum['action'] = a
                            datum['subaction'] = sa
                            datum['camera_id'] = c-1
                            datum['source'] = 'h36m'
                            datum['camera'] = camera_dict

                            box = _infer_box(datum['joints_3d_camera'], camera_dict, 0)
                            center = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))
                            scale = ((box[2] - box[0]) / 200.0, (box[3] - box[1]) / 200.0)
                            datum['center'] = center
                            datum['scale'] = scale
                            datum['box'] = box
                            box_with_confidence = box.tolist()
                            box_with_confidence.append(1.0)
                            
                            
                            metadatum = {
                                'filename': f'{subject_folder}/action_name/images/{int(camera_name):02d}/{i:06d}.jpg',  # Adjust the image path format if needed
                                'height': height,
                                'width': width,
                                'annots': [
                                    {
                                        'personID': 0,
                                        'bbox': box_with_confidence,
                                        'center':center,
                                        'keypoints': pose2d_confidence[i, joint_idx, :].tolist(),
                                        'keypoints3d':pose3d_camera_confidence[i, joint_idx, :].tolist(),
                                        'isKeyframe': False
                                        }
                                    ],
                                    'isKeyframe': False
                                    }
                            
                            subdir_out_folder = osp.join(json_folder, subject_folder, action_name)
                            os.makedirs(subdir_out_folder, exist_ok=True)

                            cam_folder = osp.join(subdir_out_folder,"images", camera_name)
                            os.makedirs(cam_folder, exist_ok=True)
                            
                            annots_folder = osp.join(subdir_out_folder,"annots", camera_name)
                            os.makedirs(annots_folder, exist_ok=True)
                            
                            image_output = f'{i:06d}.jpg'
                            image_filepath = osp.join(cam_folder, image_output) 
                            shutil.copy(source_path, image_filepath)
                            json_filename = f"{i:06d}.json"
                            json_filepath = osp.join(annots_folder, json_filename)
                            
                            with open(json_filepath, 'w') as json_file:
                                json.dump(metadatum, json_file)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('h36m_root', type=str, help='path to human3.6m dataset')
    parser.add_argument('output_folder', type=str, help='output root directory')
    args = parser.parse_args()
    main(args.h36m_root, args.output_folder)

