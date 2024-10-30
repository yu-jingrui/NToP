subjects=("anastasia" "aosilan" "arslan" "barlas" "barry" "camilo" "dannier" "gaoxing" "huajiangtao5" "joseph" "kamal_ejaz" "kemal" "lihongyun" "maria" "natacha" "quyuanning" "rabbi" "rivera" "songyujie" "sunyuxing" "wuwenyan" "zhanghao" "zhangziyu" "zhuna2")

for sub in "${subjects[@]}"; do

    config_path="/path/to/NToP/humannerf/configs/human_nerf/genebody/$sub/single_gpu_1.yaml"

    python bulkrender_genebody.py --cfg $config_path
done
