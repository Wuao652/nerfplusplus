import torch
# import torch.nn as nn
import torch.optim
import torch.distributed
# from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing
import numpy as np
import os
# from collections import OrderedDict
# from ddp_model import NerfNet
import time
# from data_loader_split import load_data_split
from carla_loader_split import load_data_split
from utils import mse2psnr, colorize_np, to8b
import imageio
from ddp_train_nerf import config_parser, setup_logger, setup, cleanup, render_single_image, create_nerf
import logging
from dcp_utils import gray2rgb, get_radiance
import cv2
logger = logging.getLogger(__package__)


def ddp_test_nerf(rank, args):
    ###### set up multi-processing
    setup(rank, args.world_size)
    ###### set up logger
    logger = logging.getLogger(__package__)
    setup_logger()

    ###### decide chunk size according to gpu memory
    if torch.cuda.get_device_properties(rank).total_memory / 1e9 > 14:
        logger.info('setting batch size according to 24G gpu')
        args.N_rand = 1024
        args.chunk_size = 8192
    else:
        logger.info('setting batch size according to 12G gpu')
        args.N_rand = 512
        args.chunk_size = 1024

    ###### create network and wrap in ddp; each process should do this
    start, models = create_nerf(rank, args)

    render_splits = [x.strip() for x in args.render_splits.strip().split(',')]
    # start testing
    for split in render_splits:
        out_dir = os.path.join(args.basedir, args.expname,
                               'render_{}_{:06d}'.format(split, start))
        if rank == 0:
            os.makedirs(out_dir, exist_ok=True)
        print(split)
        ###### load data and create ray samplers; each process should do this
        ray_samplers = load_data_split(args.datadir, args.scene, split, try_load_min_depth=args.load_min_depth)
        for idx in range(len(ray_samplers)):
            ### each process should do this; but only main process merges the results
            fname = '{:06d}.png'.format(idx)
            if ray_samplers[idx].img_path is not None:
                fname = os.path.basename(ray_samplers[idx].img_path)

            if os.path.isfile(os.path.join(out_dir, fname)):
                logger.info('Skipping {}'.format(fname))
                continue

            time0 = time.time()
            ret = render_single_image(rank, args.world_size, models, ray_samplers[idx], args.chunk_size)
            dt = time.time() - time0
            if rank == 0:    # only main process should do this
                logger.info('Rendered {} in {} seconds'.format(fname, dt))

                # only save last level
                im = ret[-1]['rgb'].numpy()
                # compute psnr if ground-truth is available
                if ray_samplers[idx].img_path is not None:
                    gt_im = ray_samplers[idx].get_img()
                    psnr = mse2psnr(np.mean((gt_im - im) * (gt_im - im)))
                    logger.info('{}: psnr={}'.format(fname, psnr))

                im = to8b(im)
                imageio.imwrite(os.path.join(out_dir, fname), im)

                # depth_img (cv2 save_img)
                depth_img_32_bit = ret[-1]['depth'].numpy() * 100.
                # normalize the image to (0, 255)
                depth_img_8_bit = cv2.normalize(depth_img_32_bit, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
                depth_img_8_bit_color = cv2.applyColorMap(depth_img_8_bit, cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(out_dir, 'full_depth_' + fname), depth_img_8_bit_color)

                # depth_img (imageio save_img)
                depth = ret[-1]['depth'].numpy()
                depth = gray2rgb(depth)
                depth = to8b(depth)
                imageio.imwrite(os.path.join(out_dir, 'depth_' + fname), depth)

                # trans_map = ret[-1]['trans_map'].numpy()
                # clear_im = get_radiance(ray_samplers[idx].get_img(), trans_map, ray_samplers[idx].get_air_light())
                # trans_map = gray2rgb(trans_map)
                # trans_map = to8b(trans_map)
                # imageio.imwrite(os.path.join(out_dir, 'trans_' + fname), trans_map)
                #
                # clear_im = to8b(clear_im)
                # imageio.imwrite(os.path.join(out_dir, 'clear_' + fname), clear_im)

                #####################################################################
                #### add by wuao : image save example ###############################
                #####################################################################

                # im = ret[-1]['fg_rgb'].numpy()
                # im = to8b(im)
                # imageio.imwrite(os.path.join(out_dir, 'fg_' + fname), im)

                # im = ret[-1]['fg_depth'].numpy()
                # im = colorize_np(im, cmap_name='jet', append_cbar=True)
                # im = to8b(im)
                # imageio.imwrite(os.path.join(out_dir, 'fg_depth_' + fname), im)

            torch.cuda.empty_cache()

    # clean up for multi-processing
    cleanup()


def test():
    parser = config_parser()
    args = parser.parse_args()
    logger.info(parser.format_values())

    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        logger.info('Using # gpus: {}'.format(args.world_size))
    torch.multiprocessing.spawn(ddp_test_nerf,
                                args=(args,),
                                nprocs=args.world_size,
                                join=True)


if __name__ == '__main__':
    setup_logger()
    test()

