# cython: language_level=3

import numpy as np
import cython
from cython.parallel import prange

cimport numpy as np
from libcpp cimport bool
from libc.float cimport FLT_MAX

cdef extern from "math.h":
    cdef float powf(float base, float exp) nogil
    cdef float fabsf(float x) nogil

ctypedef np.float32_t FLOAT_t
FLOAT = np.float32


cdef FLOAT_t clipf(FLOAT_t a, FLOAT_t lb, FLOAT_t ub) nogil:
    return min(max(a, lb), ub)


cdef int clipi(int a, int lb, int ub) nogil:
    return min(max(a, lb), ub)


@cython.wraparound(False)
def avg_pool_2d(FLOAT_t[:, ::1] x, int kernel_size):
    cdef int h = x.shape[0]
    cdef int w = x.shape[1]
    cdef int h_out = h // kernel_size
    cdef int w_out = w // kernel_size
    cdef int i, j, ii, jj

    result = np.zeros((h_out, w_out), dtype=FLOAT)
    cdef FLOAT_t[:, ::1] result_view = result

    cdef FLOAT_t tmp = 0.0
    for i in prange(h_out, nogil=True):
        for j in range(w_out):
            tmp = 0.0
            for ii in range(kernel_size):
                for jj in range(kernel_size):
                    tmp = tmp + x[i * kernel_size + ii, j * kernel_size + jj]
            tmp = tmp / <FLOAT_t>(kernel_size * kernel_size)
            result_view[i, j] = tmp

    return result


@cython.wraparound(False)
def bilinear_interpolate_2d(FLOAT_t[:, :, ::1] x, int h_out, int w_out):
    cdef int h_in = x.shape[0]
    cdef int w_in = x.shape[1]
    cdef int channels = x.shape[2]
    cdef FLOAT_t scale_h, scale_w, i_r, j_r, tmp_i1, tmp_i2, ri1, ri2, rj1, rj2
    cdef int i, j, c, i1, i2, j1, j2

    scale_h = <FLOAT_t>h_in / <FLOAT_t>h_out
    scale_w = <FLOAT_t>w_in / <FLOAT_t>w_out

    result = np.zeros((h_out, w_out, channels), dtype=FLOAT)
    cdef FLOAT_t[:, :, ::1] result_view = result

    for i in prange(h_out, nogil=True):
        for j in range(w_out):
            i_r = clipf((<FLOAT_t>i + 0.5) * scale_h - 0.5, 0, h_in - 1)
            j_r = clipf((<FLOAT_t>j + 0.5) * scale_w - 0.5, 0, w_in - 1)
            ri2 = i_r % 1.0
            ri1 = 1.0 - ri2
            rj2 = j_r % 1.0
            rj1 = 1.0 - rj2

            i1 = <int>i_r
            i2 = min(i1 + 1, h_in - 1)
            j1 = <int>j_r
            j2 = min(j1 + 1, w_in - 1)

            for c in range(channels):
                tmp_i1 = rj1 * x[i1, j1, c] + rj2 * x[i1, j2, c]
                tmp_i2 = rj1 * x[i2, j1, c] + rj2 * x[i2, j2, c]
                result_view[i, j, c] = ri1 * tmp_i1 + ri2 * tmp_i2

    return result


@cython.wraparound(False)
def align_layer(FLOAT_t[:, ::1] ref_img, FLOAT_t[:, ::1] alt_img, FLOAT_t[:, :, ::1] prev_align,
                int tile_size, int search_step, bool use_l2):
    cdef int h, w, n_ty, n_tx, y0, x0, yy, xx, dy, dx, x1, y1, x2, y2, i, j, half_tile_size
    cdef FLOAT_t min_dist, tmp_dist
    h = ref_img.shape[0]
    w = ref_img.shape[1]
    half_tile_size = tile_size // 2
    n_ty = h // half_tile_size - 1
    n_tx = w // half_tile_size - 1

    align = np.zeros((n_ty, n_tx, 2), dtype=FLOAT)  # align_y, align_x
    cdef FLOAT_t[:, :, ::1] align_view = align

    for i in prange(n_ty, nogil=True):
        for j in range(n_tx):
            min_dist = FLT_MAX
            y0 = i * half_tile_size
            x0 = j * half_tile_size
            align_view[i, j, 0] = prev_align[i, j, 0]
            align_view[i, j, 1] = prev_align[i, j, 1]

            for dy in range(-search_step, search_step + 1):
                for dx in range(-search_step, search_step + 1):
                    y1 = y0 + <int>prev_align[i, j, 0] + dy
                    x1 = x0 + <int>prev_align[i, j, 1] + dx
                    y2 = y1 + tile_size
                    x2 = x1 + tile_size
                    if y1 < 0 or x1 < 0 or h <= y2 or w <= x2:
                        continue

                    tmp_dist = 0.
                    for yy in range(tile_size):
                        for xx in range(tile_size):
                            if use_l2:
                                tmp_dist = tmp_dist + powf(ref_img[y0 + yy, x0 + xx] - alt_img[y1 + yy, x1 + xx], 2)
                            else:
                                tmp_dist = tmp_dist + fabsf(ref_img[y0 + yy, x0 + xx] - alt_img[y1 + yy, x1 + xx])

                    if tmp_dist < min_dist:
                        min_dist = tmp_dist
                        align_view[i, j, 0] = y1 - y0
                        align_view[i, j, 1] = x1 - x0

    return align
