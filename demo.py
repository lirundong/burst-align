# -*- coding: utf-8 -*-

from glob import glob
from time import time

import rawpy as rp
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def align(burst, ref_idx, pyramid_cfg, tile_cfg):
    """

    Args:
        burst:
        ref_idx:
        pyramid_cfg (list of dict): each level with bellow keys
            - downsample: shrink ratio w.r.t. previous level
        tile_cfg (list of dict):
            - shape: tile shape, e.g. [16, 16]
            - search_region: step_negative, step_positive w.r.t. tile origin, e.g. [-4, 4]

    Returns:

    """
    # 1. prepare image pyramid
    pyramids = [[img, ] for img in burst]
    for i, img in enumerate(burst):
        pyramid = pyramids[i]
        prev_img = pyramid[-1]
        for cfg in pyramid_cfg:
            downsample = cfg["downsample"]
            img = F.avg_pool2d(prev_img, downsample)
            pyramid.append(img)
            prev_img = img

    # 2. search alignment for tiles
    ref_pyramid = pyramids.pop(ref_idx)
    n_burst = len(pyramids)
    _, _, h, w = ref_pyramid[0].shape
    h_t0, w_t0 = tile_cfg[0]["shape"]
    n_ty = h // (h_t0 // 2) - 1  # num of tiles on y-axis
    n_tx = w // (w_t0 // 2) - 1  # num og tiles on x-axis
    alignment = torch.zeros((n_burst, 2, n_ty, n_tx))  # (dy, dx) on `channel` dim

    for n, pyramid in enumerate(pyramids):
        # compute alignment from last pyramid level
        for i in reversed(range(len(pyramid_cfg))):
            ref_img = ref_pyramid[i][0, 0]
            alt_img = pyramid[i][0, 0]
            assert ref_img.shape == alt_img.shape
            h, w = ref_img.shape
            h_t, w_t = tile_cfg[i]["shape"]
            step_n, step_p = tile_cfg[i]["search_region"]
            n_ty = h // (h_t // 2) - 1
            n_tx = w // (w_t // 2) - 1
            current_align = torch.zeros((1, 2, n_ty, n_tx))
            if i == len(pyramid_cfg) - 1:
                prev_align = torch.zeros((1, 2, n_ty, n_tx))
            else:
                downsample = pyramid_cfg[i + 1]["downsample"]
                prev_align = F.interpolate(prev_align, size=(n_ty, n_tx), mode="bilinear", align_corners=False) \
                             * downsample

            # iter though tiles
            for y in range(n_ty):
                for x in range(n_tx):
                    # search within [step_n, step_p]
                    min_dist = torch.finfo(torch.float32).max
                    argmin = None

                    y0 = y * (h_t // 2)
                    x0 = x * (w_t // 2)
                    for dy in range(step_n, step_p + 1):
                        for dx in range(step_n, step_p + 1):
                            y1 = y0 + dy + int(prev_align[0, 0, y, x].item())
                            x1 = x0 + dx + int(prev_align[0, 1, y, x].item())

                            if y1 < 0 or x1 < 0 or h <= y1 + h_t or w <= x1 + w_t:
                                continue

                            if i > 0:
                                dist = (ref_img[y0: y0+h_t, x0: x0+w_t] - alt_img[y1: y1+h_t, x1: x1+w_t]).norm()
                            else:
                                dist = (ref_img[y0: y0+h_t, x0: x0+w_t] - alt_img[y1: y1+h_t, x1: x1+w_t]).abs().sum()
                            if dist < min_dist:
                                min_dist = dist
                                argmin = y1 - y0, x1 - x0

                    if argmin is not None:
                        current_align[0, 0, y, x] = argmin[0]  # dy
                        current_align[0, 1, y, x] = argmin[1]  # dx
                    else:
                        # raise RuntimeError("WTF?")
                        current_align[..., y, x] = float("NaN")

            if i > 0:
                prev_align = current_align
            else:
                alignment[n, ...] = current_align

    return alignment


if __name__ == "__main__":
    raw_list = sorted(glob("./data/burst/*.dng"))
    ref_idx = 0
    n_img = 2
    pyramid_cfg = [
        {"downsample": 4},
        {"downsample": 4},
        {"downsample": 4},
    ]
    tile_cfg = [
        {"shape": [16, 16], "search_region": [-4, 4]},
        {"shape": [16, 16], "search_region": [-4, 4]},
        {"shape": [16, 16], "search_region": [-4, 4]},
    ]

    burst = []
    images = []
    for f_path in raw_list[:n_img]:
        with rp.imread(f_path) as raw:
            raw_img = raw.raw_image.copy().astype(np.float32)
            h, w = raw_img.shape
            raw_img = torch.from_numpy(raw_img).reshape(1, 1, h, w)
            raw_img = F.avg_pool2d(raw_img, 2)  # average RGGB Bayer
            color_img = raw.postprocess()
            burst.append(raw_img)
            images.append(color_img)

    t0 = time()
    alignment = align(burst, ref_idx, pyramid_cfg, tile_cfg)
    t1 = time()

    print(f"one burst ({len(burst)} images) time: {t1 - t0:.3f}s")

    f, axs = plt.subplots(n_img, 2)

    ax = axs[0, 0]
    ref_img = images.pop(ref_idx)
    ax.imshow(ref_img)
    ax.set_title("Reference Frame")

    for i, (img, a) in enumerate(zip(images, alignment)):
        frame_i = i + 1
        ax1 = axs[frame_i, 0]
        ax1.imshow(img)
        ax1.set_title(f"Frame {frame_i}")

        ax2 = axs[frame_i, 1]
        a = a.numpy()
        # dx = a[0, ...]
        # dy = a[1, ...]
        # h, w, _ = img.shape
        # yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        # ax2.quiver(xx, yy, dx, dy)
        a = a.sum(axis=0)
        a = (a - a.min()) / (a.max() - a.min())
        ax2.imshow(a)
        ax2.set_title(f"Alignment {frame_i}")

    f.tight_layout()
    f.savefig("./data/demo.png")
