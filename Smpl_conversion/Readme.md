# SMPL to SMPL-X Conversion

## Introduction

The conversion from SMPL (Simple Human Pose and Shape Model) to SMPL-X involves extending the original model to include additional features such as hands and feet. This process is essential for more realistic and detailed human character representations in computer graphics and animation.

## Background

### SMPL (Simple Human Pose and Shape Model)

[SMPL](https://smpl.is.tue.mpg.de) is a widely used model for representing human body shape and pose. It provides a parametric representation that allows for easy manipulation of body parameters to generate realistic human-like characters. It has a total of 24 Smpl Joints

### SMPL-X

[SMPL-X](https://smpl-x.is.tue.mpg.de) is an extension of SMPL that includes extra joints for hands and feet. This extension enhances the model's capabilities, making it suitable for a broader range of applications, including detailed hand and foot animations. A total of 144 joints are aligned with the Smpl-X model

## Conversion Process

The conversion from SMPL to SMPL-X typically involves modifying the model's parameters and structure to accommodate the additional joints for hands and feet.

### Steps:
1. **Setup Requirements**

Install [mesh](https://github.com/MPI-IS/mesh.git)

Start by cloning the SMPL-X repo:

    git clone https://github.com/vchoutas/smplx.git
    
Run the following command to install all necessary requirements
        
    pip install -r requirements.txt
    
Install the Torch Trust Region optimizer by following the instructions [here](https://github.com/vchoutas/torch-trust-ncg.git)

Install loguru

Install open3d

Install omegaconf

2. **Creation of Motion-Pose Files**
   
   First, break the motion into a set of pose .obj files. Depending on how the SMPL-* parameters are stored this code will differ..

3. **Config File:**
   
   To run the transfer_model utility you will require a .yaml config file, which can point to the location the output .obj files have been saved. Use the templates in config_files in the root of this repository..

4. **Conversion Step:**
   
    python -m transfer_model --exp-cfg config_files/smpl2smplx_as.yaml


