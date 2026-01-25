import pycolmap
from pathlib import Path
import imageio
from graphics_utils import pose_spherical
import numpy as np


class COLMAP:
    """
    Initalize the pipeline
    image_dir: directory path for images
    output_dir: directory to create path
    """

    def __init__(self, image_path, output_path, device):
        self.image_path = image_path
        self.output_path = Path(output_path)
        self.db_path = Path(image_path).joinpath("database.db")
        self.db_path.touch()
        self.camera_intrinscs = {}
        self.device = device

    def get_reconstruction(self):
        pycolmap.extract_features(str(self.db_path), str(self.image_path))
        pycolmap.match_exhaustive(str(self.db_path))
        reconstruction = pycolmap.incremental_mapping(
            str(self.db_path), str(self.image_path), str(self.image_path)
        )[0]
        return reconstruction

    """
        Returns:
        images: All images [N, H, W, 3 or 4]
        poses: All camera poses [N, 4, 4]
        render_poses: Novel view poses [N_render, 4, 4]
        hwf: [H, W, focal]
    """

    def get_nerf_data(self):
        reconstruction = self.get_reconstruction()
        focal_length = 0
        for camera_id in reconstruction.cameras:
            focal_length = reconstruction.camera(camera_id).focal_length_x
        all_images = []
        poses = []
        render_poses = []
        for image in reconstruction.images.values():
            im_data = imageio.imread(image.name)
            all_images.append(im_data)
            if not image.has_frame_ptr():
                print("img not registered")
                continue
            w2c = image.cam_from_world()
            c2w = w2c.inverse()
            poses.append[c2w]
        H, W = all_images[0].shape[:2]
        render_poses = np.stack(
            [
                pose_spherical(angle, -30.0, 4.0)
                for angle in np.linspace(-180, 180, 40 + 1)[:-1]
            ],
            dim=0,
        )
        return all_images, poses, render_poses, [H, W, focal_length]


if __name__ == "__main__":
    colmap = COLMAP("./testData", "./data/", pycolmap.Device.cuda)
    all_images, poses, render_poses, hwf = colmap.get_nerf_data()
    print(all_images)
    print(poses)
    print(render_poses)
