import os
import numpy as np
import imageio
import logging
from dehaze_nerf_sample_ray_split import RaySamplerSingleImage
import glob

def find_files(dir, exts):
    if os.path.isdir(dir):
        # types should be ['*.png', '*.jpg']
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    else:
        return []

def load_data_split(basedir, scene, split, skip=1, try_load_min_depth=True, only_img_files=False):
    # intrinsics files
    intrinsics_files = np.array([[224.06641222710715, 0.0, 320.0, 0.0],
                                [0.0, 224.06641222710715, 240.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

    # pose files
    # load gt pose files
    gt_pose_path = os.path.join(basedir, 'gt', scene, 'poses.npy')
    gt_pose_files = np.load(gt_pose_path)
    # load hazy pose files
    hazy_pose_path = os.path.join(basedir, 'hazy', scene, 'poses.npy')
    hazy_pose_files = np.load(hazy_pose_path)
    print(gt_pose_path)
    print(f"pose_files shape : {gt_pose_files.shape}")
    print(hazy_pose_path)
    print(f"pose_files shape : {hazy_pose_files.shape}")

    # check if the poses of gt_img and hazy_img are the same
    assert (gt_pose_files == hazy_pose_files).all()

    pose_files = gt_pose_files.copy()
    del gt_pose_files
    del hazy_pose_files
    cam_cnt = pose_files.shape[0]

    # img files
    # load gt_img files
    gt_img_files = find_files(
        os.path.join(basedir, 'gt', scene),
        exts=['*.png', '*.jpg']
    )
    if len(gt_img_files) > 0:
        print(f"load {len(gt_img_files)} ground-truth image files")
        gt_img_files = gt_img_files[::skip]
        assert (len(gt_img_files) == cam_cnt)
    else:
        gt_img_files = [None, ] * cam_cnt

    # load hazy_img files
    hazy_img_files = find_files(
        os.path.join(basedir, 'hazy', scene),
        exts=['*.png', '*.jpg']
    )

    if len(hazy_img_files) > 0:
        print(f"load {len(hazy_img_files)} hazy image files")
        hazy_img_files = hazy_img_files[::skip]
        assert (len(hazy_img_files) == cam_cnt)
    else:
        hazy_img_files = [None, ] * cam_cnt

    mask_files = [None, ] * cam_cnt
    mindepth_files = [None, ] * cam_cnt

    # pre-process the pose
    xy = pose_files[:, :-1, -1]
    # array([ -52.07071, -230.9998 ], dtype=float32)
    pose_files[:, :-1, -1] = pose_files[:, :-1, -1] - np.mean(xy, 0)

    scale_factor = 10.0
    pose_files[:, :, -1] /= scale_factor

    t = np.tile(
        np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        (cam_cnt, 1, 1))

    pose_files = np.concatenate((pose_files, t), axis=1)

    np.random.seed(34)
    i_choice = np.random.choice(cam_cnt, cam_cnt, False)
    i_split = [list(range(i, i + cam_cnt // 3)) for i in range(0, cam_cnt, cam_cnt // 3)]
    train_idx, validate_idx, test_idx = [i_choice[s] for s in i_split]

    if split == 'train':
        gt_img_files = [gt_img_files[i] for i in train_idx]
        hazy_img_files = [hazy_img_files[i] for i in train_idx]
        mask_files = [mask_files[i] for i in train_idx]
        mindepth_files = [mindepth_files[i] for i in train_idx]
        pose_files = pose_files[train_idx]
        cam_cnt = pose_files.shape[0]
    elif split == 'validation':
        gt_img_files = [gt_img_files[i] for i in validate_idx]
        hazy_img_files = [hazy_img_files[i] for i in validate_idx]
        mask_files = [mask_files[i] for i in validate_idx]
        mindepth_files = [mindepth_files[i] for i in validate_idx]
        pose_files = pose_files[validate_idx]
        cam_cnt = pose_files.shape[0]
    elif split == 'test':
        gt_img_files = [gt_img_files[i] for i in test_idx]
        hazy_img_files = [hazy_img_files[i] for i in test_idx]
        mask_files = [mask_files[i] for i in test_idx]
        mindepth_files = [mindepth_files[i] for i in test_idx]
        pose_files = pose_files[test_idx]
        cam_cnt = pose_files.shape[0]
    else:
        print('fatal error!')

    H, W = 480, 640

    print(f"Dataloader Type : {split}")
    print(f"Number of gt_img files : {len(gt_img_files)}")
    print(f"Number of hazy_img files : {len(hazy_img_files)}")
    print(f"Number of pose files : {pose_files.shape}")
    print(f"Image size : {H} * {W}")
    print(f"Preprocess facter : {scale_factor}")

    # create ray samplers
    ray_samplers = []
    for i in range(cam_cnt):
        intrinsics = intrinsics_files
        pose = pose_files[i]
        # read max depth
        max_depth = None

        # check if the names of gt_img and hazy_img are the same
        assert (os.path.basename(gt_img_files[i]) == os.path.basename(hazy_img_files[i]))

        ray_samplers.append(RaySamplerSingleImage(H=H, W=W, intrinsics=intrinsics, c2w=pose,
                                                  gt_img_path=gt_img_files[i],
                                                  hazy_img_path=hazy_img_files[i],
                                                  mask_path=mask_files[i],
                                                  min_depth_path=mindepth_files[i],
                                                  max_depth=max_depth))

    return ray_samplers


if __name__ == "__main__":
    basedir = './data/carla_data'
    scene = '9actors'
    split = 'train'
    try_load_min_depth = True
    only_img_files = False
    load_data_split(
        basedir,
        scene,
        split,
        try_load_min_depth,
        only_img_files
    )
