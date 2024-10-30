[EasyMoCap](https://chingswy.github.io/easymocap-public-doc/quickstart/quickstart.html) is an open-source toolbox designed for markerless human motion capture from RGB videos. This project offers a wide range of motion capture methods across various settings.

In this setup, two major methods have been tested:
- Using `mocap.py` (Motion Capture)
- Using `mv1p.py` (Multi-view Single Person)

Prior to EasyMocap utilization, it is important to generate certain datset files:
- Camera parameter files `intri.yml` and `extri.yml`

    - For Human3.6M dataset, the available camera parameters needs to be adjusted to EasyMocap format:

        ```bash
        cd /path/to/NToP/Easymocap/ntop/
        H36M_ROOT='/path/to/human36m/dataset/root/directory/'
        OUTPUT_DIR='/path/to/output/directory/'
        python pre_h36m_camera.py $H36M_ROOT $OUTPUT_DIR
        ```


    - For Genebody camera parameters, run the following command

        ```bash
        GB_SUB='/path/to/genebody/subject/'
        python pre_genebody_camera.py $GB_SUB
        ```

- Keypoint files (.json format)
    
    - For Human3.6M dataset, the keypoint files are extracted from the XML annotation files:

        ```bash
        H36M_ROOT='/path/to/human36m/dataset/root/directory/'
        OUTPUT_DIR='/path/to/output/directory/'
        python extract_h36m_keypoints.py $H36M_ROOT $OUTPUT_DIR
        ```

    - For Genebody dataset, run the following command:
            
        ```bash
        GB_SUB='/path/to/genebody/subject/'
        python apps/preprocess/extract_keypoints.py $GB_SUB --subs 01 10 19 28 37 46 --mode yolo-hrnet
        ```

Once the Keypoint and Camera Files are generated, create a new annots.npy file with updated camera values and image locations.

To generate the new `annots.npy` file, run the following:

```bash
cd ..
# see comments in file and change accordingly
python ntop/auto_annots.py
```

Then enter the word "annots" and respective "subject" as input.

Once the annots.npy is generated, simply run the command to execute SMPL Generation using EasyMocap:

```bash
# see comments in file and change accordingly
python ntop/auto_mocap.py
```

Then enter `mocap` or `mv1p` as input and generate the SMPL parameters.

Lastly, HumanNerf training file allows .npy files for manipulation, therefore, make internal script change & Run the following command:

```bash
# see comments in file and change accordingly
python ntop/json_npy_conversion.py
```

Now the SMPL files for the respective dataset of Human3.6m, Genebody are generated, converted and ready to used for NeRF model training
