"""
graphics_utils.py

All graphics-related utilities for NeRF:
- Ray generation
- Camera transformations
- Volume rendering
- Sampling strategies
- Data loading
"""

import numpy as np
import torch
import json
import os
import imageio


# ============================================================================
# CAMERA AND RAY GENERATION
# ============================================================================


def get_rays(H, W, focal, c2w, device="cpu"):
    """
    Generate rays for all pixels in an image.

    Args:
        H, W: Image height and width
        focal: Focal length in pixels
        c2w: Camera-to-world matrix [3, 4] or [4, 4]

    Returns:
        rays_o: Ray origins [H, W, 3]
        rays_d: Ray directions [H, W, 3]
    """
    # Create pixel coordinate grid
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing="xy"
    )
    i = i.t()  # Transpose to [H, W]
    j = j.t()
    i = i.to(device)
    j = j.to(device)

    # Directions in camera space
    # X: right, Y: up, Z: backward (camera looks down -Z)
    dirs = torch.stack(
        [
            (i - W * 0.5) / focal,  # X component
            -(j - H * 0.5) / focal,  # Y component (flip for image coords)
            -torch.ones_like(i),  # Z component (looking forward)
        ],
        dim=-1,
    )
    dirs = dirs.to(device)

    # Transform directions from camera space to world space
    # rays_d = dirs @ R^T where R is the rotation part of c2w
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    rays_d = rays_d.to(device)

    # Ray origins are just the camera position (same for all rays)
    rays_o = c2w[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    """NumPy version of get_rays."""
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )

    dirs = np.stack(
        [(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -np.ones_like(i)], axis=-1
    )

    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], axis=-1)
    rays_o = np.broadcast_to(c2w[:3, -1], rays_d.shape)

    return rays_o, rays_d


# ============================================================================
# SAMPLING ALONG RAYS
# ============================================================================


def sample_along_rays(
    rays_o, rays_d, near, far, N_samples, perturb=False, lindisp=False
):
    """
    Sample points along rays.

    Args:
        rays_o: Ray origins [N_rays, 3]
        rays_d: Ray directions [N_rays, 3]
        near: Near bound for sampling
        far: Far bound for sampling
        N_samples: Number of samples per ray
        perturb: If True, add stratified randomness
        lindisp: If True, sample linearly in disparity (1/depth)

    Returns:
        pts: Sampled 3D points [N_rays, N_samples, 3]
        z_vals: Distances along rays [N_rays, N_samples]
    """
    N_rays = rays_o.shape[0]

    # Create evenly spaced samples in [0, 1]
    t_vals = torch.linspace(0.0, 1.0, N_samples, device=rays_o.device)

    # Map to [near, far]
    if not lindisp:
        # Linear in depth
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        # Linear in disparity (inverse depth)
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)

    # Expand to all rays
    z_vals = z_vals.expand([N_rays, N_samples])

    # Stratified sampling - add randomness within bins
    if perturb:
        # Get bin midpoints
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)

        # Random uniform samples in each bin
        t_rand = torch.rand(z_vals.shape, device=z_vals.device)
        z_vals = lower + (upper - lower) * t_rand

    # Compute 3D points: point = origin + distance * direction
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    return pts, z_vals


def sample_pdf(bins, weights, N_samples, det=False):
    """
    Sample from a piecewise-uniform PDF (for hierarchical sampling).

    Args:
        bins: Bin edges [N_rays, N_bins]
        weights: Weights for each bin [N_rays, N_bins]
        N_samples: Number of samples to draw
        det: If True, deterministic sampling

    Returns:
        samples: Sampled distances [N_rays, N_samples]
    """
    # Normalize weights to get PDF
    weights = weights + 1e-5  # Prevent division by zero
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)

    # Compute CDF
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, N_samples, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=cdf.device)

    # Invert CDF (find where u falls in CDF)
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], dim=-1)

    # Interpolate
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


# ============================================================================
# VOLUME RENDERING
# ============================================================================


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0.0, white_bkgd=False):
    """
    Transform network outputs to RGB and other maps via volume rendering.

    Args:
        raw: Network predictions [N_rays, N_samples, 4] (RGB + density)
        z_vals: Sample distances [N_rays, N_samples]
        rays_d: Ray directions [N_rays, 3]
        raw_noise_std: Std of noise to add to density (regularization)
        white_bkgd: If True, composite on white background

    Returns:
        rgb_map: Rendered RGB [N_rays, 3]
        disp_map: Disparity map [N_rays]
        acc_map: Accumulated opacity [N_rays]
        weights: Sample weights [N_rays, N_samples]
        depth_map: Depth map [N_rays]
    """

    # Compute distances between adjacent samples
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # Last distance is infinity
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)

    # Multiply by ray direction norm to get real world distance
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Extract RGB (apply sigmoid to get [0, 1] range)
    rgb = torch.sigmoid(raw[..., :3])

    # Add noise to density predictions (training only)
    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn(raw[..., 3].shape, device=raw.device) * raw_noise_std

    # Compute alpha (opacity) from density
    # alpha = 1 - exp(-density * distance)
    density = raw[..., 3]
    alpha = 1.0 - torch.exp(-torch.relu(density + noise) * dists)

    # Compute weights using transmittance
    # T_i = prod(1 - alpha_j) for j < i
    transmittance = torch.cumprod(
        torch.cat(
            [torch.ones((alpha.shape[0], 1), device=alpha.device), 1.0 - alpha + 1e-10],
            dim=-1,
        ),
        dim=-1,
    )[:, :-1]

    weights = alpha * transmittance

    # Compute final outputs
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [N_rays, 3]
    depth_map = torch.sum(weights * z_vals, dim=-1)
    disp_map = 1.0 / torch.max(
        1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, dim=-1)
    )
    acc_map = torch.sum(weights, dim=-1)

    # Composite onto white background if requested
    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


# ============================================================================
# DATA LOADING
# ============================================================================


def load_blender_data(basedir, half_res=False, testskip=1):
    """
    Load Blender synthetic dataset.

    Args:
        basedir: Base directory containing the dataset
        half_res: If True, load images at half resolution
        testskip: Load every N-th test/val image

    Returns:
        images: All images [N, H, W, 3 or 4]
        poses: All camera poses [N, 4, 4]
        render_poses: Novel view poses [N_render, 4, 4]
        hwf: [H, W, focal]
        i_split: [i_train, i_val, i_test] indices
    """
    splits = ["train", "val", "test"]
    metas = {}

    # Load metadata
    for s in splits:
        with open(os.path.join(basedir, f"transforms_{s}.json"), "r") as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]

    # Load images and poses
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []

        skip = 1 if s == "train" or testskip == 0 else testskip

        for frame in meta["frames"][::skip]:
            fname = os.path.join(basedir, frame["file_path"] + ".png")
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame["transform_matrix"]))

        # Normalize to [0, 1]
        imgs = (np.array(imgs) / 255.0).astype(np.float32)
        poses = np.array(poses).astype(np.float32)

        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    # Create split indices
    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, axis=0)
    poses = np.concatenate(all_poses, axis=0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    # Generate render poses (circle around scene)
    render_poses = torch.stack(
        [
            pose_spherical(angle, -30.0, 4.0)
            for angle in np.linspace(-180, 180, 40 + 1)[:-1]
        ],
        dim=0,
    )

    # Downsample if requested
    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.0

        imgs_half = np.zeros((imgs.shape[0], H, W, imgs.shape[3]))
        for i, img in enumerate(imgs):
            imgs_half[i] = (
                torch.nn.functional.interpolate(
                    torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0),
                    size=(H, W),
                    mode="area",
                )
                .squeeze(0)
                .permute(1, 2, 0)
                .numpy()
            )
        imgs = imgs_half

    return imgs, poses, render_poses, [H, W, focal], i_split


def pose_spherical(theta, phi, radius):
    """
    Generate a spherical camera pose.

    Args:
        theta: Azimuth angle in degrees
        phi: Elevation angle in degrees
        radius: Distance from origin

    Returns:
        c2w: Camera-to-world matrix [4, 4]
    """

    def trans_t(t):
        return torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]],
            dtype=torch.float32,
        )

    def rot_phi(phi):
        return torch.tensor(
            [
                [1, 0, 0, 0],
                [0, np.cos(phi), -np.sin(phi), 0],
                [0, np.sin(phi), np.cos(phi), 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )

    def rot_theta(th):
        return torch.tensor(
            [
                [np.cos(th), 0, -np.sin(th), 0],
                [0, 1, 0, 0],
                [np.sin(th), 0, np.cos(th), 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )

    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w

    # Blender to NeRF coordinate system
    c2w = (
        torch.tensor(
            [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )
        @ c2w
    )

    return c2w


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def to8b(x):
    """Convert float image to 8-bit."""
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def img2mse(x, y):
    """Compute MSE between two images."""
    return torch.mean((x - y) ** 2)


def mse2psnr(mse):
    """Convert MSE to PSNR."""
    return -10.0 * torch.log10(mse)
