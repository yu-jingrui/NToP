import os
import numpy as np
import cv2
import numpy as np

class FileStorage(object):
    def __init__(self, filename, isWrite=False):
        version = cv2.__version__
        self.major_version = int(version.split('.')[0])
        self.second_version = int(version.split('.')[1])

        if isWrite:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.fs = open(filename, 'w')
            self.fs.write('%YAML:1.0\r\n')
            self.fs.write('---\r\n')
        else:
            assert os.path.exists(filename), filename
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        self.isWrite = isWrite

    def __del__(self):
        if self.isWrite:
            self.fs.close()
        else:
            cv2.FileStorage.release(self.fs)

    def _write(self, out):
        self.fs.write(out+'\r\n')

    def write(self, key, value, dt='mat'):
        if dt == 'mat':
            self._write('{}: !!opencv-matrix'.format(key))
            self._write('  rows: {}'.format(value.shape[0]))
            self._write('  cols: {}'.format(value.shape[1]))
            self._write('  dt: d')
            self._write('  data: [{}]'.format(', '.join(['{:.6f}'.format(i) for i in value.reshape(-1)])))
        elif dt == 'list':
            self._write('{}:'.format(key))
            for elem in value:
                self._write('  - "{}"'.format(elem))
        elif dt == 'int':
            self._write('{}: {}'.format(key, value))

    def read(self, key, dt='mat'):
        if dt == 'mat':
            output = self.fs.getNode(key).mat()
        elif dt == 'list':
            results = []
            n = self.fs.getNode(key)
            for i in range(n.size()):
                val = n.at(i).string()
                if val == '':
                    val = str(int(n.at(i).real()))
                if val != 'none':
                    results.append(val)
            output = results
        elif dt == 'int':
            output = int(self.fs.getNode(key).real())
        else:
            raise NotImplementedError
        return output

    def close(self):
        self.__del__(self)


def write_camera(camera, path):
    intri_name = os.path.join(path, 'intri.yml')
    extri_name = os.path.join(path, 'extri.yml')
    intri = FileStorage(intri_name, True)
    extri = FileStorage(extri_name, True)
    results = {}
    camnames = [key_.split('.')[0] for key_ in camera.keys()]
    intri.write('names', camnames, 'list')
    extri.write('names', camnames, 'list')
    for key_, val in camera.items():
        if key_ == 'basenames':
            continue
        key = key_.split('.')[0]
        intri.write('K_{}'.format(key), val['K'])
        intri.write('dist_{}'.format(key), val['dist'])
        if 'H' in val.keys() and 'W' in val.keys():
            intri.write('H_{}'.format(key), val['H'], dt='int')
            intri.write('W_{}'.format(key), val['W'], dt='int')
        assert val['R'].shape == (3, 3), f"{val['R'].shape} must == (3, 3)"
        if 'Rvec' not in val.keys():
            val['Rvec'] = cv2.Rodrigues(val['R'])[0]
        extri.write('R_{}'.format(key), val['Rvec'])
        extri.write('Rot_{}'.format(key), val['R'])
        extri.write('T_{}'.format(key), val['T'])


def copy_images(input_folder, output_folder, subfolders):
    os.makedirs(output_folder, exist_ok=True)
    for subfolder in subfolders:
        subfolder_path = os.path.join(input_folder, subfolder)
        if os.path.isdir(subfolder_path):
            file_list = os.listdir(subfolder_path)
            file_list.sort()
            counter = 0
            for file_name in file_list:
                if file_name.lower().endswith((".jpg", ".png")):
                    new_file_name = f"{counter:06d}.jpg"
                    old_file_path = os.path.join(subfolder_path, file_name)
                    new_file_path = os.path.join(output_folder, subfolder, new_file_name)
                    os.makedirs(os.path.join(output_folder, subfolder), exist_ok=True)
                    os.rename(old_file_path, new_file_path)
                    counter += 1
    print("Images copied and saved in the output folder with the same structure as the original.")


def copy_masks(input_folder, output_folder, subfolders):
    os.makedirs(output_folder, exist_ok=True)
    for subfolder in subfolders:
        subfolder_path = os.path.join(input_folder, subfolder)
        if os.path.isdir(subfolder_path):
            file_list = os.listdir(subfolder_path)
            file_list.sort()
            counter = 0
            for file_name in file_list:
                if file_name.lower().endswith((".jpg", ".png")):
                    new_file_name = f"mask{counter:06d}.png"
                    old_file_path = os.path.join(subfolder_path, file_name)
                    new_file_path = os.path.join(output_folder, subfolder, new_file_name)
                    os.makedirs(os.path.join(output_folder, subfolder), exist_ok=True)
                    os.rename(old_file_path, new_file_path)
                    counter += 1
    print("Masks copied and saved in the output folder with the same structure as the original.")


def main(subject_path, debug=False):
    root_path = subject_path
    cam_list = ['01', '10', '19', '28', '37', '46']
    annot_path = os.path.join(root_path, 'annots.npy')
    data = np.load(annot_path, allow_pickle=True).item()
    cam_data = data['cams']
    cameras = {}
    for cam_num in cam_list:
        cam = cam_data[cam_num]
        K = cam['K']
        D = cam['D'].reshape(1, 5)
        E = np.linalg.inv(np.vstack((np.hstack((cam['c2w_R'], cam['c2w_T'])), [0, 0, 0, 1])))
        c2w_R = E[:3, :3]
        c2w_T = np.array([E[:3, 3]]).reshape(3, 1)
        rvec, _ = cv2.Rodrigues(c2w_R)
        if debug:
            print(f"Camera Number: {cam_num}")
            print(f"K: {K}")
            print(f"rvec: {rvec}")
            print(f"D: {D}")
            print(f"c2w_R: {c2w_R}")
            print(f"c2w_T: {c2w_T}")
        cameras[cam_num] = {
            'K': K,
            'dist': D,
            'R': c2w_R,
            'T': c2w_T,
            'E': E,
            'rvec': rvec
        }

    image_input_folder = os.path.join(root_path, 'image')
    image_output_folder = os.path.join(root_path, 'images')
    copy_images(image_input_folder, image_output_folder, cam_list)

    mask_input_folder = os.path.join(root_path, 'mask')
    mask_output_folder = os.path.join(root_path, 'masks')
    copy_masks(mask_input_folder, mask_output_folder, cam_list)

    output = root_path
    write_camera(cameras, output)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='path to genebody dataset subject')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(args.path, args.debug)
