subjects=("S1" "S5" "S6" "S7" "S8" "S9" "S11")
actions=("Sitting" "SittingDown" "Smoking" "Waiting" "WalkDog" "Walking" "WalkTogether" "Sitting" "SittingDown" "Smoking" "Waiting" "WalkDog" "Walking" "WalkTogether")
for sub in "${subjects[@]}"; do
    for ac in "${actions[@]}"; do

        config_path="/path/to/NToP/humannerf/configs/human_nerf/h36m/$sub/actions/adventure_multiview_$ac.yaml"
	
        python bulkrender.py --cfg $config_path
    done
done
