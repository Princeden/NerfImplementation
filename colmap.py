import pycolmap
from pathlib import Path


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

    def extract_intrinsics(self):
        pycolmap.extract_features(str(self.db_path), str(self.image_path))
        pycolmap.match_exhaustive(str(self.db_path))
        reconstruction = pycolmap.incremental_mapping(
                str(self.db_path),
                str(self.image_path),
                str(self.image_path)
        )[0]
        print(reconstruction)
        db = pycolmap.Database.open(self.db_path)
        focal_length = 0
        for camera_id in reconstruction.cameras:
            focal_length = reconstruction.camera(camera_id).focal_length_x
        #pycolmap.incremental_mapping(str(self.db_path), str(self.image_path), str(self.image_path))
        for image in reconstruction.images.values():
            if not image.has_frame_ptr():
                print("img not regist")
                continue
            w2c = image.cam_from_world()
            c2w = w2c.inverse()
            print(c2w)
            print(c2w.todict())

        # reconstruction = pycolmap.incremental_mapping(
        #     database_path=self.db_path,
        #     image_path=self.image_path,
        #     output_path=self.output_path,
        # )
        # print(reconstruction)
        return

    def get_nerf_json(self):
        return


if __name__ == "__main__":
    colmap = COLMAP("./testData", "./data/", pycolmap.Device.cuda)
    colmap.extract_intrinsics()
