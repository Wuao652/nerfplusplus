import numpy as np
from collections import OrderedDict
import torch
import cv2
import imageio
import matplotlib.pyplot as plt
from dcp_utils import idx2sub, gray2rgb
from dcp_utils import get_dark_channel, \
    get_atmosphere, \
    get_transmission_estimate, \
    get_laplacian, \
    get_radiance

import scipy
import scipy.sparse.linalg
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
                       img_path=None,
                       img_clear_path=None,
                       resolution_level=1,
                       mask_path=None,
                       min_depth_path=None,
                       max_depth=None):
        super().__init__()
        self.W_orig = W
        self.H_orig = H
        self.intrinsics_orig = intrinsics
        self.c2w_mat = c2w

        self.img_path = img_path
        self.img_clear_path = img_clear_path
        self.mask_path = mask_path
        self.min_depth_path = min_depth_path
        self.max_depth = max_depth

        self.resolution_level = -1
        self.set_resolution_level(resolution_level)

        # # window size of the image patch using in dcp
        # self.win_size = 15
        # self.omega = 0.95
        # # dark_channel image of the input hazy image. [H, W]
        # self.dark_channel = get_dark_channel(self.img.reshape(H, W, -1), self.win_size)
        # # air_light. [1, 3]
        # self.air_light = get_atmosphere(self.img.reshape(H, W, -1), self.dark_channel)
        # # coarse transmission map. [H, W]
        # self.coarse_t = get_transmission_estimate(self.img.reshape(H, W, -1),
        #                                           self.air_light,
        #                                           self.omega,
        #                                           self.win_size)
        # self.coarse_t = self.coarse_t.reshape(-1)   # [H*W]

        # self.img [H*W, 3]
        # self.coarse_t [H*W]
        # self.rays_o [H*W, 3]
        # self.rays_d [H*W, 3]
        # self.depth [H*W]

    def set_resolution_level(self, resolution_level):
        if resolution_level != self.resolution_level:
            self.resolution_level = resolution_level
            self.W = self.W_orig // resolution_level
            self.H = self.H_orig // resolution_level
            self.intrinsics = np.copy(self.intrinsics_orig)
            self.intrinsics[:2, :3] /= resolution_level
            # only load image at this time
            if self.img_path is not None:
                self.img = imageio.imread(self.img_path).astype(np.float32) / 255.
                self.img = cv2.resize(self.img, (self.W, self.H), interpolation=cv2.INTER_AREA)
                self.img = self.img.reshape((-1, 3))
            else:
                self.img = None

            # load the clear image
            if self.img_clear_path is not None:
                self.img_clear = imageio.imread(self.img_clear_path).astype(np.float32) / 255.
                self.img_clear = cv2.resize(self.img_clear, (self.W, self.H), interpolation=cv2.INTER_AREA)
                self.img_clear = self.img_clear.reshape((-1, 3))
            else:
                self.img_clear = None

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
    def get_img(self):
        if self.img is not None:
            return self.img.reshape((self.H, self.W, 3))
        else:
            return None

    def get_clear_img(self):
        if self.img_clear is not None:
            return self.img_clear.reshape((self.H, self.W, 3))
        else:
            return None


    # def random_sample(self, N_rand):
    #     """
    #     randomly sample N_rand image patches, find the corresponding rgb, coarse_t, ray_o,
    #     ray_d, depth, mask, min_depth. Then, stack the pixels contain in the image patch
    #     and store them in an OrderedDict.
    #     :param N_rand: number of image patches to be cast.
    #     :return:
    #         ret: [H, W] np array of the dark channel image of the input hazy image.
    #
    #     """
    #     img_size = self.H * self.W  # 307200
    #     ind_mat = np.arange(img_size).reshape(self.H, self.W)
    #
    #     win_rad = 1
    #     max_num_neigh = (2 * win_rad + 1) ** 2
    #
    #     # selected_pixels = [0, 4487, 306558, 307199]
    #     selected_pixels = np.random.choice(self.H*self.W, size=(N_rand,), replace=False)
    #     num_pixel_in_patches = []
    #     selected_patch_indices = []
    #
    #     for idx in selected_pixels:
    #         i, j = idx2sub(self.H, self.W, idx)
    #         H_min = max(0, i - win_rad)
    #         H_max = min(self.H, i + win_rad)
    #         W_min = max(0, j - win_rad)
    #         W_max = min(self.W, j + win_rad)
    #
    #         win_inds = ind_mat[H_min: H_max + 1, W_min: W_max + 1]
    #         win_inds = win_inds.reshape(-1)
    #         num_neigh = win_inds.size
    #         num_pixel_in_patches.append(num_neigh)
    #         selected_patch_indices.append(win_inds)
    #
    #     selected_patch_indices = np.concatenate(selected_patch_indices, -1)
    #
    #     rays_o = self.rays_o[selected_patch_indices, :]
    #     rays_d = self.rays_d[selected_patch_indices, :]
    #     depth = self.depth[selected_patch_indices]
    #     coarse_t = self.coarse_t[selected_patch_indices]
    #     if self.img is not None:
    #         rgb = self.img[selected_patch_indices, :]
    #     else:
    #         rgb = None
    #     if self.min_depth is not None:
    #         min_depth = self.min_depth[selected_patch_indices]
    #     else:
    #         min_depth = 1e-4 * np.ones_like(rays_d[..., 0])
    #
    #     ret = OrderedDict([
    #         ('ray_o', rays_o),
    #         ('ray_d', rays_d),
    #         ('depth', depth),
    #         ('rgb', rgb),
    #         ('coarse_t', coarse_t),
    #         ('min_depth', min_depth),
    #         ('img_name', self.img_path),
    #         ('num_pixel_in_patches', num_pixel_in_patches)
    #     ])
    #     # return torch tensors
    #     for k in ret:
    #         if isinstance(ret[k], np.ndarray):
    #             ret[k] = torch.from_numpy(ret[k])
    #
    #     return ret

    def get_all(self):
        if self.min_depth is not None:
            min_depth = self.min_depth
        else:
            min_depth = 1e-4 * np.ones_like(self.rays_d[..., 0])

        ret = OrderedDict([
            ('ray_o', self.rays_o),
            ('ray_d', self.rays_d),
            ('depth', self.depth),
            ('rgb', self.img),
            ('rgb_clear', self.img_clear),
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

        if self.img is not None:
            rgb = self.img[select_inds, :]          # [N_rand, 3]
        else:
            rgb = None

        if self.img_clear is not None:
            rgb_clear = self.img_clear[select_inds, :]      # [N_rand, 3]
        else:
            rgb_clear = None

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
            ('rgb', rgb),
            ('rgb_clear', rgb_clear),
            ('mask', mask),
            ('min_depth', min_depth),
            ('img_name', self.img_path),
            ('img_clear_name', self.img_clear_path)
        ])
        # return torch tensors
        for k in ret:
            if isinstance(ret[k], np.ndarray):
                ret[k] = torch.from_numpy(ret[k])

        return ret

if __name__ == "__main__":
    H, W = 480, 640
    intrinsics = np.array([[224.06641222710715, 0.0, 320.0, 0.0],
                           [0.0, 224.06641222710715, 240.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

    pose = np.array([[-7.5747770e-01,  2.1615420e-01, -6.1603969e-01,  3.7919426e-01],
                     [ 6.5286100e-01,  2.5079149e-01, -7.1475595e-01,  4.3173981e-01],
                     [ 1.1102230e-16, -9.4360000e-01, -3.3108762e-01,  2.0000000e-01],
                     [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]], dtype=np.float32)

    img_files = './data/carla_data/hazy/9actors/r_00147.png'
    img_clear_files = './data/carla_data/gt/9actors/r_00147.png'

    raysampler = RaySamplerSingleImage(H=H, W=W, intrinsics=intrinsics, c2w=pose,
                          img_path=img_files,
                          img_clear_path=img_clear_files,
                          mask_path=None,
                          min_depth_path=None,
                          max_depth=None)
    ret = raysampler.random_sample(512)

    # Lambda = 0.0001
    # img = raysampler.img.reshape(H, W, -1)
    # print(raysampler.air_light)
    # print(raysampler.air_light.shape)
    # L = get_laplacian(img)
    # A = L + Lambda * scipy.sparse.eye(H * W)
    # b = Lambda * raysampler.coarse_t.reshape(H, W).T.reshape(-1)
    # x = scipy.sparse.linalg.spsolve(A, b)
    # transmission = x.reshape((W, H)).T
    # radiance = get_radiance(img, transmission, raysampler.air_light)
    #
    #
    # plt.figure()
    # plt.imshow(img)
    # plt.figure()
    # plt.imshow(gray2rgb(raysampler.coarse_t.reshape(H, W)))
    # plt.figure()
    # plt.imshow(gray2rgb(transmission))
    # plt.figure()
    # plt.imshow(radiance)
    # plt.show()