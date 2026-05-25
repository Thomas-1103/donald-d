import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

from .core import (
    normalize_matrix,
    compute_structure_tensor_2d,
    tensors2d_to_eigs,
    ellipse_from_tensor_2d,
)


def plot_2d_matrix_with_ellipses(
    M,
    tokens=None,
    gradient_sigma=0.9,
    tensor_smooth_sigma=0.9,
    ellipse_subsample=1,
    ellipse_scale=0.6,
    show_heatmap=True,
    normalize_rows=True,
    normalize_cols=False,
    normalize_global=False,
    output_file=None,
):
    if normalize_global:
        M = normalize_matrix(M, normalize_global=True)
    elif normalize_rows or normalize_cols:
        M = normalize_matrix(
            M,
            normalize_rows=normalize_rows,
            normalize_cols=normalize_cols,
        )

    Ldim, Tdim = M.shape

    Tgrid = compute_structure_tensor_2d(
        M,
        gradient_sigma=gradient_sigma,
        tensor_smooth_sigma=tensor_smooth_sigma,
    )

    evals, evecs, anis, V1 = tensors2d_to_eigs(Tgrid)

    anchor_colors = [
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
    ]

    positions = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors_loop = list(anchor_colors) + [anchor_colors[0]]

    cmap4 = LinearSegmentedColormap.from_list(
        "four_anchor_custom",
        list(zip(positions, colors_loop)),
        N=256,
    )

    vx = V1[..., 0]
    vy = V1[..., 1]

    angles = np.arctan2(vy, vx)
    angles_mod = np.mod(angles, np.pi)
    f = angles_mod / np.pi

    rgba_img = cmap4(f)
    rgb_img = rgba_img[..., :3]

    gamma = 1.8
    anis_clipped = np.clip(anis, 0.0, 1.0)
    shade = anis_clipped ** gamma

    tile_rgb = rgb_img * shade[..., None] + (1.0 - shade[..., None])
    tile_rgb = np.clip(tile_rgb, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(12, 6))

    if show_heatmap:
        ax.imshow(
            tile_rgb,
            origin="lower",
            aspect="equal",
            interpolation="nearest",
            extent=[-0.5, Tdim - 0.5, -0.5, Ldim - 0.5],
            alpha=1.0,
        )

    for yi in range(0, Ldim, ellipse_subsample):
        for xi in range(0, Tdim, ellipse_subsample):
            tensor = Tgrid[yi, xi]
            base_color = tuple(tile_rgb[yi, xi].tolist())

            ellipse_from_tensor_2d(
                (xi, yi),
                tensor,
                scale=ellipse_scale,
                alpha=0.9,
                ax=ax,
                edgecolor="k",
                facecolor=base_color,
            )

    if tokens is not None:
        ax.set_xticks(np.arange(Tdim))
        ax.set_xticklabels(tokens, rotation=90)

    ax.set_ylabel("Layer")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, bbox_inches="tight", dpi=600)

    plt.show()

    return {
        "M": M,
        "Tgrid": Tgrid,
        "evals": evals,
        "evecs": evecs,
        "anis": anis,
        "V1": V1,
    }