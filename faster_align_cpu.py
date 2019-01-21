# -*- coding: utf-8 -*-

from glob import glob
from time import time

import rawpy as rp
import numpy as np
import matplotlib.pyplot as plt

import align


def align_burst(burst, ref_idx, pyramid_cfg, tile_cfg):
    """ A multi-thread implementation of L2/L1 alignment algorithm.

    This algorithm is described in HDR+ paper: http://www.hdrplusdata.org/

    Args:
        burst (list of np.array): input raw burst images;
        ref_idx (int): index of reference frame, eg. 0;
        pyramid_cfg (list of dict): each level with bellow keys
            - downsample: shrink ratio w.r.t. previous level;
        tile_cfg (list of dict):
            - size (int): tile shape, eg. 16;
            - search_step (int): search step size w.r.t. tile origin, eg. 4;

    Returns:
        A list of alignment map w.r.t. each frame in `burst`.

    """
    pyramids = [[img, ] for img in burst]
    for i, img in enumerate(burst):
        pyramid = pyramids[i]
        prev_img = pyramid[-1]
        for cfg in pyramid_cfg:
            downsample = cfg["downsample"]
            img = align.avg_pool_2d(prev_img, downsample)
            pyramid.append(img)
            prev_img = img

    ref_pyramid = pyramids.pop(ref_idx)
    alignment = []
    for n, pyramid in enumerate(pyramids):
        # compute alignment from last pyramid level
        for i in reversed(range(len(pyramid_cfg))):
            ref_img = ref_pyramid[i]
            alt_img = pyramid[i]
            h, w = ref_img.shape
            tile_size = tile_cfg[i]["size"]
            search_step = tile_cfg[i]["search_step"]
            n_ty = h // (tile_size // 2) - 1
            n_tx = w // (tile_size // 2) - 1
            if i == len(pyramid_cfg) - 1:
                prev_align = np.zeros((n_ty, n_tx, 2), dtype=np.float32)
            else:
                downsample = pyramid_cfg[i + 1]["downsample"]
                prev_align = align.bilinear_interpolate_2d(prev_align, n_ty, n_tx) * downsample

            use_l2 = i != 0
            current_align = align.align_layer(ref_img, alt_img, prev_align, tile_size, search_step, use_l2)

            if i > 0:
                prev_align = current_align
            else:
                alignment.append(current_align)

    return alignment


if __name__ == "__main__":
    raw_list = sorted(glob("./data/burst/*.dng"))
    ref_idx = 0
    n_img = 10
    pyramid_cfg = [
        {"downsample": 4},
        {"downsample": 4},
        {"downsample": 4},
        # {"downsample": 4},
    ]
    tile_cfg = [
        {"size": 16, "search_step": 4},
        {"size": 16, "search_step": 4},
        {"size": 16, "search_step": 4},
        # {"size": 16, "search_step": 4},
    ]

    burst = []
    images = []
    for f_path in raw_list[:n_img]:
        with rp.imread(f_path) as raw:
            raw_img = raw.raw_image.copy().astype(np.float32)
            raw_img = align.avg_pool_2d(raw_img, 2)
            color_img = raw.postprocess()
            burst.append(raw_img)
            images.append(color_img)

    t0 = time()
    alignment = align_burst(burst, ref_idx, pyramid_cfg, tile_cfg)
    t1 = time()

    print(f"one burst ({len(burst)} images) time: {t1 - t0:.3f}s")

    f, axs = plt.subplots(n_img, 2, figsize=(5 * 3, 5 * n_img))

    ax = axs[0, 0]
    ref_img = images.pop(ref_idx)
    ax.imshow(ref_img)
    ax.set_title("Reference Frame")
    ax = axs[0, 1]
    ax.axis("off")

    for i, (img, a) in enumerate(zip(images, alignment)):
        frame_i = i + 1
        ax1 = axs[frame_i, 0]
        ax1.imshow(img)
        ax1.set_title(f"Frame {frame_i}")

        ax2 = axs[frame_i, 1]
        dx = a[..., 1]
        dy = a[..., 0]
        h, w = dx.shape
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        ax2.quiver(xx, yy, dx[::-1, ...], dy[::-1, ...])
        ax2.set_title(f"Alignment {frame_i}")

    f.tight_layout()
    f.savefig(f"./data/demo_{n_img}_imgs.png")
