from configs import cfg

class DatasetArgs(object):
    dataset_attrs = {}

    subjects = ['313', '315', '377', '386', '387', '390', '392', '393', '394', 'female-3-casual', 'female-4-casual', 'male-3-casual', 'male-4-casual', 'male-5-outdoor', 'female-6-plaza']
    h36m_subs = ['S1', 'S5', 'S7', 'S6', 'S8', 'S9', 'S11']
    h36m_actions = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo', 'Posing', 'Purchases',
                    'Sitting', 'SittingDown', 'Smoking', 'Waiting', 'WalkDog', 'Walking', 'WalkTogether']
    genebody_subs = ['ahha', 'alejandro', 'anastasia', 'barry', 'gaoxing', 'huajiangtao5', 'joseph', 'kamal_ejaz', 'kemal', 'lihongyun', 'natacha', 'quyuanning', 'rabbi', 'rivera', 'songyujie', 'sunyuxing', 'wuwenyan', 'zhanghao', 'zhangziyu', 'zhuna2', 'aosilan','arslan','barlas','camilo','dannier','xujiarui','zhanghongwei','zhonglantai']
    people_snap_subs = ['female-3-casual', 'female-4-casual', 'male-3-casual', 'male-4-casual', 'male-5-outdoor']
    
    if cfg.category == 'human_nerf' and cfg.task == 'zju_mocap':
        for sub in subjects:
            dataset_attrs.update({
                f"zju_{sub}_train": {
                    "dataset_path": f"dataset/zju_mocap/{sub}",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                },
                f"zju_{sub}_test": {
                    "dataset_path": f"dataset/zju_mocap/{sub}", 
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap'
                },
            })

    if cfg.category == 'human_nerf' and cfg.task == 'wild':
        for subs in h36m_subs:
            dataset_attrs.update({
            f"monocular_{subs}_train": {
                "dataset_path": f"dataset/wild/{subs}",
                "keyfilter": cfg.train_keyfilter,
                "ray_shoot_mode": cfg.train.ray_shoot_mode,
            },
            f"monocular_{subs}_test": {
                "dataset_path": f"dataset/wild/{subs}",
                "keyfilter": cfg.test_keyfilter,
                "ray_shoot_mode": 'image',
                "src_type": 'wild'
            },
            })

    if cfg.category == 'human_nerf' and cfg.task == 'people_snapshot':
        for subs in subjects:
            dataset_attrs.update({
                f"{subs}_train": {
                    "dataset_path": f"dataset/people_snapshot/{subs}",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                },
                f"{subs}_test": {
                    "dataset_path": f"dataset/people_snapshot/{subs}", 
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap'
                },
            })

    if cfg.category == 'human_nerf' and cfg.task == 'h36m':
        for subs in h36m_subs:
            dataset_attrs.update({
                f"h36m_{subs}_train": {
                    "dataset_path": f"dataset/h36m/{subs}",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                },
                f"h36m_{subs}_test": {
                    "dataset_path": f"dataset/h36m/{subs}",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'h36m'
                },
            })
    if cfg.category == 'human_nerf' and cfg.task == 'h36m_multiview':
        for subs in h36m_subs:
            dataset_attrs.update({
                f"h36m_{subs}_train": {
                    "dataset_path": f"dataset/h36m_multiview/{subs}",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                },
                f"h36m_{subs}_test": {
                    "dataset_path": f"dataset/h36m_multiview/{subs}",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'h36m'
                },
            })
    if cfg.category == 'human_nerf' and cfg.task == 'h36m_multiview' and 'action' in cfg.keys():
        for sub in h36m_subs:
            for ac in h36m_actions:
                dataset_attrs.update({
                    f"h36m_{sub}_{ac}_train": {
                        "dataset_path": f"dataset/h36m_multiview/{sub}/{ac}",
                        "keyfilter": cfg.train_keyfilter,
                        "ray_shoot_mode": cfg.train.ray_shoot_mode,
                    },
                    f"h36m_{sub}_{ac}_test": {
                        "dataset_path": f"dataset/h36m_multiview/{sub}/{ac}",
                        "keyfilter": cfg.test_keyfilter,
                        "ray_shoot_mode": 'image',
                        "src_type": 'h36m'
                    },
                })
    if cfg.category == 'human_nerf' and cfg.task == 'genebody':
        for subs in genebody_subs:
            dataset_attrs.update({
                f'genebody_{subs}_train': {
                    'dataset_path': f'dataset/genebody/{subs}',
                    'keyfilter': cfg.train_keyfilter,
                    'ray_shoot_mode': cfg.train.ray_shoot_mode,
                },
                f'genebody_{subs}_test': {
                    'dataset_path': f'dataset/genebody/{subs}',
                    'keyfilter': cfg.test_keyfilter,
                    'ray_shoot_mode': 'image',
                    'src_type': 'genebody',
                },
            })

    @staticmethod
    def get(name):
        attrs = DatasetArgs.dataset_attrs[name]
        return attrs.copy()
