import math
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator


def normalize_rows_to_unit(M, eps=1e-12):
    M = np.asarray(M, dtype=float)
    row_min = M.min(axis=1, keepdims=True)
    row_max = M.max(axis=1, keepdims=True)
    denom = (row_max - row_min) + eps
    M_norm = (M - row_min) / denom
    return np.clip(M_norm, 0.0, 1.0)


def normalize_columns_to_unit(M, eps=1e-12):
    M = np.asarray(M, dtype=float)
    col_min = M.min(axis=0, keepdims=True)
    col_max = M.max(axis=0, keepdims=True)
    denom = (col_max - col_min) + eps
    M_norm = (M - col_min) / denom
    return np.clip(M_norm, 0.0, 1.0)


def normalize_global_to_unit(M, eps=1e-12):
    M = np.asarray(M, dtype=float)
    global_min = M.min()
    global_max = M.max()
    denom = (global_max - global_min) + eps
    M_norm = (M - global_min) / denom
    return np.clip(M_norm, 0.0, 1.0)


def normalize_matrix(M, normalize_rows=False, normalize_cols=False, normalize_global=False, eps=1e-12):
    M2 = np.asarray(M, dtype=float).copy()
    if normalize_global:
        return normalize_global_to_unit(M2, eps=eps)
    if normalize_rows:
        M2 = normalize_rows_to_unit(M2, eps=eps)
    if normalize_cols:
        M2 = normalize_columns_to_unit(M2, eps=eps)
    return M2


def collapse_embeddings_to_matrix(embeddings_stack):
    arr = np.asarray(embeddings_stack)
    if arr.ndim == 3:
        a, b, c = arr.shape
        if a <= 64 and b <= 64 and c >= 64:
            if a == 12:
                M = arr.mean(axis=2)
            elif b == 12:
                M = arr.transpose(1, 0, 2).mean(axis=2)
            elif c == 12:
                M = arr.mean(axis=0)
            else:
                M = arr.mean(axis=2)
        else:
            M = arr.mean(axis=2)
    else:
        raise ValueError(f"Unsupported embeddings shape: {arr.shape}")
    return np.asarray(M, dtype=float)


def compute_structure_tensor_2d(M, gradient_sigma=0.9, tensor_smooth_sigma=0.9, regularize=1e-12):
    gy, gx = np.gradient(M)
    if gradient_sigma and gradient_sigma > 0:
        gy = gaussian_filter(gy, sigma=gradient_sigma)
        gx = gaussian_filter(gx, sigma=gradient_sigma)

    Jxx = gx * gx
    Jxy = gx * gy
    Jyy = gy * gy

    if tensor_smooth_sigma and tensor_smooth_sigma > 0:
        Jxx = gaussian_filter(Jxx, sigma=tensor_smooth_sigma)
        Jxy = gaussian_filter(Jxy, sigma=tensor_smooth_sigma)
        Jyy = gaussian_filter(Jyy, sigma=tensor_smooth_sigma)

    Ldim, Tdim = M.shape
    Tgrid = np.zeros((Ldim, Tdim, 2, 2), dtype=float)
    Tgrid[..., 0, 0] = Jxx
    Tgrid[..., 0, 1] = Jxy
    Tgrid[..., 1, 0] = Jxy
    Tgrid[..., 1, 1] = Jyy
    Tgrid += regularize * np.eye(2).reshape((1, 1, 2, 2))
    return Tgrid


def compute_fraction_along_row(Tgrid, eps=1e-12):
    Jxx = Tgrid[..., 0, 0]
    Jyy = Tgrid[..., 1, 1]
    trace = Jxx + Jyy

    frac_cell = Jxx / (trace + eps)

    row_num = np.sum(Jxx, axis=1)
    row_den = np.sum(trace, axis=1)
    frac_row_weighted = row_num / (row_den + eps)
    frac_row_mean = np.mean(frac_cell, axis=1)

    return frac_cell, frac_row_weighted, frac_row_mean


def tensors2d_to_eigs(Tgrid, eps=1e-12):
    Ldim, Tdim = Tgrid.shape[:2]
    N = Ldim * Tdim
    A = Tgrid.reshape((N, 2, 2))

    vals, vecs = np.linalg.eigh(A)
    vals = vals[:, ::-1]
    vecs = vecs[:, :, ::-1]

    evals = vals.reshape((Ldim, Tdim, 2))
    evecs = vecs.reshape((Ldim, Tdim, 2, 2))

    l1 = evals[..., 0]
    l2 = evals[..., 1]
    anis = (l1 - l2) / (l1 + l2 + eps)

    V1 = evecs[..., :, 0]
    norms = np.linalg.norm(V1, axis=-1, keepdims=True) + 1e-12
    V1 = V1 / norms

    return evals, evecs, anis, V1


def build_2d_interpolators(V1, anis, x_spacing=1.0, y_spacing=1.0, origin=(0.0, 0.0)):
    Ldim, Tdim = anis.shape
    x_coords = origin[0] + x_spacing * np.arange(Tdim)
    y_coords = origin[1] + y_spacing * np.arange(Ldim)

    vec_interp_raw = RegularGridInterpolator(
        (y_coords, x_coords),
        V1,
        bounds_error=False,
        fill_value=(0.0, 0.0),
    )
    anis_interp_raw = RegularGridInterpolator(
        (y_coords, x_coords),
        anis,
        bounds_error=False,
        fill_value=0.0,
    )

    def vec_interp_xy(pt_xy):
        arr = np.asarray(pt_xy)
        if arr.ndim == 1 and arr.size == 2:
            pt_yx = (arr[1], arr[0])
            return np.array(vec_interp_raw(pt_yx))
        pts = np.array([(p[1], p[0]) for p in arr])
        return vec_interp_raw(pts)

    def anis_interp_xy(pt_xy):
        arr = np.asarray(pt_xy)
        if arr.ndim == 1 and arr.size == 2:
            pt_yx = (arr[1], arr[0])
            return float(np.asarray(anis_interp_raw(pt_yx)).item())
        pts = np.array([(p[1], p[0]) for p in arr])
        return anis_interp_raw(pts)

    return vec_interp_xy, anis_interp_xy, (x_coords, y_coords)


def ellipse_from_tensor_2d(center_xy, tensor2x2, scale=1.0, alpha=0.4, ax=None, edgecolor="k", facecolor="cyan"):
    vals, vecs = np.linalg.eigh(tensor2x2)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    a = math.sqrt(max(vals[0], 0.0)) * scale
    b = math.sqrt(max(vals[1], 0.0)) * scale
    vx, vy = vecs[:, 0]
    angle = math.degrees(math.atan2(vy, vx))

    if ax is None:
        ax = plt.gca()

    ell = Ellipse(
        xy=(center_xy[0], center_xy[1]),
        width=2 * a,
        height=2 * b,
        angle=angle,
        edgecolor=edgecolor,
        facecolor=facecolor,
        alpha=alpha,
    )
    ax.add_patch(ell)
    return ell