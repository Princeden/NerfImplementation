from data import DataLoader
import numpy as np


def generate_rays_simple(H, W, focal, c2w):
    i, j = np.meshgrid(np.arrange(W), np.arrange(H), indexing="xy")
    dirs = np.stack(
        [(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -np.ones_like(i)], -1
    )
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], rays_d.shape)
    return rays_o, rays_d


if __name__ == "__main__":
    testing = True
    if testing:
        data_loader = DataLoader()
        blender_data = data_loader.get_blender_data("chair")
        print("Loaded Data")
        print(blender_data.keys())
        h = blender_data["images"][0]
        # print(h)
