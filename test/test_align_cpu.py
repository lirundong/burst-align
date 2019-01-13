# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn.functional as F
import numpy as np

import align


def test_avg_pool_2d():
    h, w = 100, 100

    for i in range(10):
        k = 2 if i % 2 == 0 else 4

        x = np.random.randn(h, w).astype(np.float32)
        y = align.avg_pool_2d(x, k)

        x_torch = torch.from_numpy(x).unsqueeze_(0).unsqueeze_(0)
        y_torch = F.avg_pool2d(x_torch, k, k)
        y_torch = y_torch.detach().numpy().squeeze()

        np.testing.assert_allclose(y, y_torch, rtol=1e-5)


def test_bilinear_interpolate_2d():
    h_in, w_in = 50, 50
    h_out, w_out = 100, 100

    for i in range(10):
        x = np.random.randn(h_in, w_in, i + 1).astype(np.float32)
        y = align.bilinear_interpolate_2d(x, h_out, w_out)

        x_torch = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze_(0)
        y_torch = F.interpolate(x_torch, size=(h_out, w_out), mode="bilinear", align_corners=False)
        y_torch = y_torch.detach().squeeze(0).numpy().transpose(1, 2, 0)

        np.testing.assert_allclose(y, y_torch, rtol=1e-5)


def test_align_layer():
    h, w = 224, 224
    tile_size = 8
    search_step = 4
    n_ty = h // (tile_size // 2) - 1
    n_tx = w // (tile_size // 2) - 1

    for i in range(10):
        use_l2 = i % 2 == 0
        ref_img = np.random.randn(h, w).astype(np.float32)
        alt_img = np.random.randn(h, w).astype(np.float32)
        prev_align = np.random.randn(n_ty, n_tx, 2).astype(np.float32)

        align_map = align.align_layer(ref_img, alt_img, prev_align, tile_size, search_step, use_l2)
