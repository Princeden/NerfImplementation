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
        pycolmap.extract_features(
            str(self.db_path),
            self.image_path,
            [],
            pycolmap.CameraMode.AUTO,
            "SIMPLE_RADIAL",
            pycolmap.ImageReaderOptions(),
            pycolmap.FeatureExtractionOptions(),
            self.device,
        )
        pycolmap.match_exhaustive(
            self.db_path,
            pycolmap.FeatureMatchingOptions(),
            pycolmap.ExhaustivePairingOptions(),
            pycolmap.TwoViewGeometryOptions(),
            self.device,
        )
        num_images = pycolmap.Database(self.db_path).num_images
        print(num_images)

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
    colmap = COLMAP("./colmapData/cup", "./data/", pycolmap.Device.cuda)
    colmap.extract_intrinsics()
