import numpy as np
from collections import OrderedDict
import torch
import cv2
import imageio
import matplotlib.pyplot as plt

########################################################################################################################
# ray batch sampling
########################################################################################################################
def get_rays_single_image(H, W, intrinsics, c2w):
    '''
    :param H: image height
    :param W: image width
    :param intrinsics: 4 by 4 intrinsic matrix
    :param c2w: 4 by 4 camera to world extrinsic matrix
    :return:
        rays_o: (H * W, 3) ray origins for each pixel
        rays_d: (H * W, 3) ray directions for each pixel
        depth: (H * W, ) z-value of the world origin in camera coordinate
    '''
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    u = u.reshape(-1).astype(dtype=np.float32) + 0.5    # add half pixel
    v = v.reshape(-1).astype(dtype=np.float32) + 0.5
    pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)

    rays_d = np.dot(np.linalg.inv(intrinsics[:3, :3]), pixels)
    rays_d = np.dot(c2w[:3, :3], rays_d)  # (3, H*W)
    rays_d = rays_d.transpose((1, 0))  # (H*W, 3)

    rays_o = c2w[:3, 3].reshape((1, 3))
    rays_o = np.tile(rays_o, (rays_d.shape[0], 1))  # (H*W, 3)

    depth = np.linalg.inv(c2w)[2, 3]
    depth = depth * np.ones((rays_o.shape[0],), dtype=np.float32)  # (H*W,)

    return rays_o, rays_d, depth


class RaySamplerSingleImage(object):
    def __init__(self, H, W, intrinsics, c2w,
                       gt_img_path=None,
                       hazy_img_path=None,
                       resolution_level=1,
                       mask_path=None,
                       min_depth_path=None,
                       max_depth=None):
        super().__init__()
        self.W_orig = W
        self.H_orig = H
        self.intrinsics_orig = intrinsics
        self.c2w_mat = c2w

        self.gt_img_path = gt_img_path
        self.hazy_img_path = hazy_img_path
        self.mask_path = mask_path
        self.min_depth_path = min_depth_path
        self.max_depth = max_depth

        self.resolution_level = -1
        self.set_resolution_level(resolution_level)

    def set_resolution_level(self, resolution_level):
        if resolution_level != self.resolution_level:
            self.resolution_level = resolution_level
            self.W = self.W_orig // resolution_level
            self.H = self.H_orig // resolution_level
            self.intrinsics = np.copy(self.intrinsics_orig)
            self.intrinsics[:2, :3] /= resolution_level
            # only load image at this time

            # load gt_image
            if self.gt_img_path is not None:
                self.gt_img = imageio.imread(self.gt_img_path).astype(np.float32) / 255.
                self.gt_img = cv2.resize(self.gt_img, (self.W, self.H), interpolation=cv2.INTER_AREA)
                self.gt_img = self.gt_img.reshape((-1, 3))
            else:
                self.gt_img = None

            # load hazy_image
            if self.hazy_img_path is not None:
                self.hazy_img = imageio.imread(self.hazy_img_path).astype(np.float32) / 255.
                self.hazy_img = cv2.resize(self.hazy_img, (self.W, self.H), interpolation=cv2.INTER_AREA)
                self.hazy_img = self.hazy_img.reshape((-1, 3))
            else:
                self.hazy_img = None

            if self.mask_path is not None:
                self.mask = imageio.imread(self.mask_path).astype(np.float32) / 255.
                self.mask = cv2.resize(self.mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                self.mask = self.mask.reshape((-1))
            else:
                self.mask = None

            if self.min_depth_path is not None:
                self.min_depth = imageio.imread(self.min_depth_path).astype(np.float32) / 255. * self.max_depth + 1e-4
                self.min_depth = cv2.resize(self.min_depth, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
                self.min_depth = self.min_depth.reshape((-1))
            else:
                self.min_depth = None

            self.rays_o, self.rays_d, self.depth = get_rays_single_image(self.H, self.W,
                                                                         self.intrinsics, self.c2w_mat)

    def get_gt_img(self):
        if self.gt_img is not None:
            return self.gt_img.reshape((self.H, self.W, 3))
        else:
            return None

    def get_hazy_img(self):
        if self.hazy_img is not None:
            return self.hazy_img.reshape((self.H, self.W, 3))
        else:
            return None

    def get_all(self):
        if self.min_depth is not None:
            min_depth = self.min_depth
        else:
            min_depth = 1e-4 * np.ones_like(self.rays_d[..., 0])

        ret = OrderedDict([
            ('ray_o', self.rays_o),
            ('ray_d', self.rays_d),
            ('depth', self.depth),
            ('gt_rgb', self.gt_img),
            ('hazy_rgb', self.hazy_img),
            ('mask', self.mask),
            ('min_depth', min_depth)
        ])
        # return torch tensors
        for k in ret:
            if ret[k] is not None:
                ret[k] = torch.from_numpy(ret[k])
        return ret

    def random_sample(self, N_rand, center_crop=False):
        '''
        :param N_rand: number of rays to be casted
        :return:
        '''
        if center_crop:
            half_H = self.H // 2
            half_W = self.W // 2
            quad_H = half_H // 2
            quad_W = half_W // 2

            # pixel coordinates
            u, v = np.meshgrid(np.arange(half_W-quad_W, half_W+quad_W),
                               np.arange(half_H-quad_H, half_H+quad_H))
            u = u.reshape(-1)
            v = v.reshape(-1)

            select_inds = np.random.choice(u.shape[0], size=(N_rand,), replace=False)

            # Convert back to original image
            select_inds = v[select_inds] * self.W + u[select_inds]
        else:
            # Random from one image
            select_inds = np.random.choice(self.H*self.W, size=(N_rand,), replace=False)

        rays_o = self.rays_o[select_inds, :]    # [N_rand, 3]
        rays_d = self.rays_d[select_inds, :]    # [N_rand, 3]
        depth = self.depth[select_inds]         # [N_rand, ]

        if self.gt_img is not None:
            gt_rgb = self.gt_img[select_inds, :]          # [N_rand, 3]
        else:
            gt_rgb = None

        if self.hazy_img is not None:
            hazy_rgb = self.hazy_img[select_inds, :]        # [N_rand, 3]
        else:
            hazy_rgb = None

        if self.mask is not None:
            mask = self.mask[select_inds]
        else:
            mask = None

        if self.min_depth is not None:
            min_depth = self.min_depth[select_inds]
        else:
            min_depth = 1e-4 * np.ones_like(rays_d[..., 0])

        ret = OrderedDict([
            ('ray_o', rays_o),
            ('ray_d', rays_d),
            ('depth', depth),
            ('gt_rgb', gt_rgb),
            ('hazy_rgb', hazy_rgb),
            ('mask', mask),
            ('min_depth', min_depth),
            ('gt_img_name', self.gt_img_path),
            ('hazy_img_name', self.hazy_img_path)
        ])
        # return torch tensors
        for k in ret:
            if isinstance(ret[k], np.ndarray):
                ret[k] = torch.from_numpy(ret[k])

        return ret

if __name__ == "__main__":
    np.random.seed(777)
    torch.manual_seed(777)
    H, W = 480, 640
    intrinsics = np.array([[224.06641222710715, 0.0, 320.0, 0.0],
                           [0.0, 224.06641222710715, 240.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

    pose = np.array([[-7.5747752e-01,  1.6614874e-01, -6.3136548e-01,  3.7920189e-01],
                     [ 6.5286124e-01,  1.9277288e-01, -7.3253727e-01,  4.3174285e-01],
                     [ 7.4505806e-09, -9.6707457e-01, -2.5449321e-01,  1.5000001e-01],
                     [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]], dtype=np.float32)


    gt_img_files = './data/carla_data/gt/9actors/r_00147.png'
    hazy_img_files = './data/carla_data/hazy/9actors/r_00147.png'

    raysampler = RaySamplerSingleImage(H=H, W=W, intrinsics=intrinsics, c2w=pose,
                          gt_img_path=gt_img_files,
                          hazy_img_path=hazy_img_files,
                          mask_path=None,
                          min_depth_path=None,
                          max_depth=None)
    gt_img = raysampler.get_gt_img()
    hazy_img = raysampler.get_hazy_img()
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(gt_img)
    # plt.subplot(122)
    # plt.imshow(hazy_img)
    # plt.show()
    ray_batch = raysampler.random_sample(N_rand=512)