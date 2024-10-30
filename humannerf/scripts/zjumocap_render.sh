actors=("313" "315" "377" "386" "387" "390" "392" "393" "394")

for ac in "${actors[@]}"; do

    config_path="/path/to/NToP/humannerf/configs/human_nerf/zju_mocap/$ac/adventure_1.yaml"

    python bulkrender_zjumocap.py --cfg $config_path
done
