"""

Code taken from the original implemn
model.py

NeRF model architecture and training logic:
- Positional encoding
- NeRF MLP
- Rendering pipeline
- Training loop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import imageio

from graphics_utils import (
    get_rays,
    sample_along_rays,
    sample_pdf,
    raw2outputs,
    load_blender_data,
    to8b,
    img2mse,
    mse2psnr,
)


# ============================================================================
# POSITIONAL ENCODING
# ============================================================================


class PositionalEncoder(nn.Module):
    """Positional encoding using sin/cos functions."""

    def __init__(self, input_dims=3, num_freqs=10, include_input=True):
        """
        Args:
            input_dims: Dimension of input (3 for positions, 3 for directions)
            num_freqs: Number of frequency bands
            include_input: If True, concatenate raw input
        """
        super().__init__()
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        self.include_input = include_input

        # Create frequency bands: [2^0, 2^1, ..., 2^(num_freqs-1)]
        self.freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)

        # Calculate output dimension
        self.out_dim = 0
        if include_input:
            self.out_dim += input_dims
        self.out_dim += input_dims * num_freqs * 2  # *2 for sin and cos

    def forward(self, x):
        """
        Apply positional encoding.

        Args:
            x: Input tensor [..., input_dims]

        Returns:
            Encoded tensor [..., out_dim]
        """
        if self.include_input:
            out = [x]
        else:
            out = []

        # Apply sin and cos at each frequency
        for freq in self.freq_bands:
            out.append(torch.sin(2.0 * np.pi * freq * x))
            out.append(torch.cos(2.0 * np.pi * freq * x))

        return torch.cat(out, dim=-1)


def get_embedder(num_freqs, input_dims=3):
    """Create positional encoder."""
    embedder = PositionalEncoder(input_dims=input_dims, num_freqs=num_freqs)
    return embedder, embedder.out_dim


# ============================================================================
# NERF MODEL
# ============================================================================


class NeRF(nn.Module):
    """
    NeRF MLP architecture.
    """

    def __init__(
        self,
        D=8,
        W=256,
        input_ch=3,
        input_ch_views=3,
        output_ch=4,
        skips=[4],
        use_viewdirs=False,
    ):
        """
        Args:
            D: Number of layers
            W: Width of each layer
            input_ch: Input channels for positions (with encoding)
            input_ch_views: Input channels for view directions (with encoding)
            output_ch: Output channels (4 for RGB + density)
            skips: Layers to add skip connections at
            use_viewdirs: If True, use view-dependent rendering
        """
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        # Position encoding layers
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)]
            + [
                nn.Linear(W, W) if i not in skips else nn.Linear(W + input_ch, W)
                for i in range(D - 1)
            ]
        )

        # View direction encoding
        if use_viewdirs:
            # Separate feature extraction and RGB prediction
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)  # Density (view-independent)
            self.rgb_linear = nn.Linear(W // 2, 3)  # RGB (view-dependent)
        else:
            # Simple output
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [..., input_ch + input_ch_views]

        Returns:
            Output tensor [..., 4] (RGB + density)
        """
        # Split input into position and view direction
        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_views], dim=-1
        )

        # Process position through MLP
        h = input_pts
        for i, layer in enumerate(self.pts_linears):
            h = F.relu(layer(h))
            if i in self.skips:
                h = torch.cat([input_pts, h], dim=-1)

        if self.use_viewdirs:
            # Density is view-independent
            alpha = self.alpha_linear(h)

            # Feature for RGB prediction
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], dim=-1)

            # Process with view direction
            for i, layer in enumerate(self.views_linears):
                h = F.relu(layer(h))

            # RGB is view-dependent
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], dim=-1)
        else:
            outputs = self.output_linear(h)

        return outputs


# ============================================================================
# RENDERING PIPELINE
# ============================================================================


def run_network(pts, viewdirs, network, embedder, embedder_dirs, netchunk=1024 * 64):
    """
    Run network on input points with batching.

    Args:
        pts: Points to query [..., N_samples, 3]
        viewdirs: Viewing directions [..., 3]
        network: NeRF model
        embedder: Position encoder
        embedder_dirs: Direction encoder
        netchunk: Batch size for network queries

    Returns:
        Network outputs [..., N_samples, 4]
    """
    # Flatten
    pts_flat = pts.reshape(-1, pts.shape[-1])

    # Encode positions
    embedded = embedder(pts_flat)

    # Encode and broadcast viewing directions
    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(pts.shape)
        input_dirs_flat = input_dirs.reshape(-1, input_dirs.shape[-1])
        embedded_dirs = embedder_dirs(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], dim=-1)

    # Run network in chunks
    outputs = []
    for i in range(0, embedded.shape[0], netchunk):
        outputs.append(network(embedded[i : i + netchunk]))
    outputs = torch.cat(outputs, dim=0)

    # Reshape back
    outputs = outputs.reshape(list(pts.shape[:-1]) + [outputs.shape[-1]])

    return outputs


def render_rays(
    ray_batch,
    network_fn,
    network_query_fn,
    N_samples,
    retraw=False,
    lindisp=False,
    perturb=0.0,
    N_importance=0,
    network_fine=None,
    white_bkgd=False,
    raw_noise_std=0.0,
):
    """
    Render rays using volumetric rendering.

    Args:
        ray_batch: [N_rays, 11] (origins, directions, near, far, viewdirs)
        network_fn: Coarse network
        network_query_fn: Function to query network
        N_samples: Number of coarse samples
        retraw: If True, return raw network outputs
        lindisp: If True, sample in disparity
        perturb: Stratified sampling parameter
        N_importance: Number of fine samples
        network_fine: Fine network
        white_bkgd: If True, white background
        raw_noise_std: Noise std for regularization

    Returns:
        Dictionary with rgb_map, disp_map, acc_map, etc.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = ray_batch[:, 6:8].reshape(-1, 1, 2)
    near, far = bounds[..., 0], bounds[..., 1]

    # Sample points along rays (coarse)
    pts, z_vals = sample_along_rays(
        rays_o, rays_d, near, far, N_samples, perturb=perturb > 0, lindisp=lindisp
    )

    # Query coarse network
    raw = network_query_fn(pts, viewdirs, network_fn)

    # Volume rendering
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std, white_bkgd
    )

    # Hierarchical sampling (fine network)
    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        # Sample more points based on coarse weights
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.0)
        )
        z_samples = z_samples.detach()

        # Combine coarse and fine samples
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)

        # Recompute points
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        # Query fine network
        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)

        # Volume rendering with fine samples
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d, raw_noise_std, white_bkgd
        )

    # Collect outputs
    ret = {"rgb_map": rgb_map, "disp_map": disp_map, "acc_map": acc_map}
    if retraw:
        ret["raw"] = raw
    if N_importance > 0:
        ret["rgb0"] = rgb_map_0
        ret["disp0"] = disp_map_0
        ret["acc0"] = acc_map_0

    return ret


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    """Render rays in smaller batches."""
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i : i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], dim=0) for k in all_ret}
    return all_ret


def render(
    H,
    W,
    focal,
    chunk=1024 * 32,
    rays=None,
    c2w=None,
    near=0.0,
    far=1.0,
    use_viewdirs=False,
    **kwargs,
):
    """
    Render full image or ray batch.

    Args:
        H, W: Image dimensions
        focal: Focal length
        chunk: Batch size
        rays: Pre-generated rays
        c2w: Camera-to-world matrix
        near, far: Depth bounds
        use_viewdirs: Use view directions

    Returns:
        rgb_map, disp_map, acc_map, extras
    """
    if c2w is not None:
        # Generate rays from camera pose
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        rays_o, rays_d = rays

    if use_viewdirs:
        viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        viewdirs = viewdirs.reshape(-1, 3).float()

    # Flatten
    rays_o = rays_o.reshape(-1, 3).float()
    rays_d = rays_d.reshape(-1, 3).float()

    # Add bounds
    near = near * torch.ones_like(rays_d[..., :1])
    far = far * torch.ones_like(rays_d[..., :1])
    rays_batch = torch.cat([rays_o, rays_d, near, far], dim=-1)

    if use_viewdirs:
        rays_batch = torch.cat([rays_batch, viewdirs], dim=-1)

    # Render
    all_ret = batchify_rays(rays_batch, chunk, **kwargs)
    if c2w is not None:
        for k in all_ret:
            k_shape = [H, W] + list(all_ret[k].shape[1:])
            all_ret[k] = all_ret[k].reshape(k_shape)

    k_extract = ["rgb_map", "disp_map", "acc_map"]
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}

    return ret_list + [ret_dict]


# ============================================================================
# TRAINING
# ============================================================================


def train():
    """Main training function."""

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    basedir = "./logs"
    expname = "lego_test"
    datadir = "./data/nerf_synthetic/lego"

    # Training params
    N_iters = 200000
    N_rand = 1024  # Batch size
    lrate = 5e-4
    lrate_decay = 250  # Decay every 250k steps

    # Model params
    netdepth = 8
    netwidth = 256
    netdepth_fine = 8
    netwidth_fine = 256

    # Rendering params
    N_samples = 64  # Coarse samples
    N_importance = 128  # Fine samples
    perturb = 1.0  # Stratified sampling
    use_viewdirs = True
    multires = 10  # Position encoding frequencies
    multires_views = 4  # Direction encoding frequencies
    raw_noise_std = 0.0
    white_bkgd = True

    # System
    chunk = 1024 * 32
    netchunk = 1024 * 64

    # Load data
    print("Loading data...")
    images, poses, render_poses, hwf, i_split = load_blender_data(
        datadir, half_res=True, testskip=8
    )
    i_train, i_val, i_test = i_split
    H, W, focal = hwf
    H, W = int(H), int(W)

    # Handle white background
    if white_bkgd:
        images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
    else:
        images = images[..., :3]

    # To torch
    images = torch.from_numpy(images).to(device)
    poses = torch.from_numpy(poses).to(device)

    print(f"Loaded {len(images)} images, {H}x{W}")
    print(f"Train: {len(i_train)}, Val: {len(i_val)}, Test: {len(i_test)}")

    # Create models
    print("Creating models...")
    embed_fn, input_ch = get_embedder(multires)
    embeddirs_fn, input_ch_views = get_embedder(multires_views)

    model = NeRF(
        D=netdepth,
        W=netwidth,
        input_ch=input_ch,
        input_ch_views=input_ch_views,
        output_ch=4,
        skips=[4],
        use_viewdirs=use_viewdirs,
    ).to(device)

    grad_vars = list(model.parameters())

    model_fine = None
    if N_importance > 0:
        model_fine = NeRF(
            D=netdepth_fine,
            W=netwidth_fine,
            input_ch=input_ch,
            input_ch_views=input_ch_views,
            output_ch=4,
            skips=[4],
            use_viewdirs=use_viewdirs,
        ).to(device)
        grad_vars += list(model_fine.parameters())

    # Network query function
    def network_query_fn(inputs, viewdirs, network):
        return run_network(inputs, viewdirs, network, embed_fn, embeddirs_fn, netchunk)

    # Optimizer
    optimizer = torch.optim.Adam(grad_vars, lr=lrate)

    # Prepare rays
    print("Preparing rays...")
    from graphics_utils import get_rays_np

    rays = np.stack(
        [get_rays_np(H, W, focal, p) for p in poses[:, :3, :4].cpu().numpy()], axis=0
    )
    rays_rgb = np.concatenate([rays, images[:, None].cpu().numpy()], axis=1)
    rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
    rays_rgb = np.stack([rays_rgb[i] for i in i_train], axis=0)
    rays_rgb = rays_rgb.reshape(-1, 3, 3)
    rays_rgb = rays_rgb.astype(np.float32)
    np.random.shuffle(rays_rgb)

    i_batch = 0

    print("Training...")
    for i in tqdm(range(N_iters)):
        # Sample random rays
        batch = rays_rgb[i_batch : i_batch + N_rand]
        batch = torch.from_numpy(batch).to(device)
        batch = batch.transpose(0, 1)
        batch_rays, target_s = batch[:2], batch[2]

        i_batch += N_rand
        if i_batch >= rays_rgb.shape[0]:
            np.random.shuffle(rays_rgb)
            i_batch = 0

        # Render
        rgb, disp, acc, extras = render(
            H,
            W,
            focal,
            chunk=chunk,
            rays=batch_rays,
            retraw=True,
            network_fn=model,
            network_query_fn=network_query_fn,
            N_samples=N_samples,
            N_importance=N_importance,
            network_fine=model_fine,
            perturb=perturb,
            raw_noise_std=raw_noise_std,
            white_bkgd=white_bkgd,
            use_viewdirs=use_viewdirs,
            near=2.0,
            far=6.0,
        )

        # Compute loss
        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if "rgb0" in extras:
            img_loss0 = img2mse(extras["rgb0"], target_s)
            loss = loss + img_loss0

        loss.backward()
        optimizer.step()

        # Learning rate decay
        decay_rate = 0.1
        decay_steps = lrate_decay * 1000
        new_lrate = lrate * (decay_rate ** (i / decay_steps))
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lrate

        # Logging
        if i % 100 == 0:
            tqdm.write(
                f"[TRAIN] Iter: {i} Loss: {loss.item():.4f} PSNR: {psnr.item():.2f}"
            )

        # Save checkpoint
        if i % 10000 == 0 and i > 0:
            path = os.path.join(basedir, expname, f"{i:06d}.tar")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(
                {
                    "iter": i,
                    "network_state_dict": model.state_dict(),
                    "network_fine_state_dict": model_fine.state_dict()
                    if model_fine
                    else None,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                path,
            )
            print(f"Saved checkpoint: {path}")

        # Render validation
        if i % 1000 == 0 and i > 0:
            img_i = np.random.choice(i_val)
            target = images[img_i]
            pose = poses[img_i, :3, :4]

            with torch.no_grad():
                rgb, disp, acc, _ = render(
                    H,
                    W,
                    focal,
                    chunk=chunk,
                    c2w=pose,
                    network_fn=model,
                    network_query_fn=network_query_fn,
                    N_samples=N_samples,
                    N_importance=N_importance,
                    network_fine=model_fine,
                    perturb=0.0,
                    raw_noise_std=0.0,
                    white_bkgd=white_bkgd,
                    use_viewdirs=use_viewdirs,
                    near=2.0,
                    far=6.0,
                )

            val_loss = img2mse(rgb, target)
            val_psnr = mse2psnr(val_loss)

            testimgdir = os.path.join(basedir, expname, "val_imgs")
            os.makedirs(testimgdir, exist_ok=True)
            imageio.imwrite(
                os.path.join(testimgdir, f"{i:06d}.png"), to8b(rgb.cpu().numpy())
            )

            print(f"[VAL] Iter: {i} PSNR: {val_psnr.item():.2f}")
    path = os.path.join(basedir, expname, "finalNerf.tar")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "iter": i,
            "network_state_dict": model.state_dict(),
            "network_fine_state_dict": model_fine.state_dict() if model_fine else None,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    print(f"Saved checkpoint: {path}")
    print("Training complete!")


if __name__ == "__main__":
    train()
