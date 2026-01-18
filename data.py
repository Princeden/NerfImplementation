import numpy as np
import os
import json
import imageio.v3 as imageio


class DataLoader:
    def __init__(self):
        self.base_dir = "./data/nerf_synthetic/"

    """
    Returns a dict of dicts
    - images (the renderings as normalized np arrays)
    - poses (the camera pose as a matrix)
    - camera_angle_x (the camera angle from the x-axis)
    """

    def get_blender_data(self, image_dir, type="train"):
        image_path = os.path.join(self.base_dir, image_dir)
        print(image_path)
        data = {}
        for file in os.listdir(image_path):
            if f"transforms_{type}.json" in file:  # get specific data type
                path_to_json = os.path.join(image_path, file)
                with open(path_to_json, "r") as json_file:  # process json
                    json_data = json.load(json_file)
                    data["camera_angle_x"] = json_data["camera_angle_x"]
                    images = []
                    poses = []
                    for frame in json_data["frames"]:
                        frame_path = os.path.join(
                            image_path, frame["file_path"][2:] + ".png"
                        )
                        images.append(imageio.imread(frame_path))
                        poses.append(np.array(frame["transform_matrix"]))
                    images = (np.asarray(images) / 255).astype(np.float32)
                    print(np.max(images))
                    data["images"] = images
                    data["poses"] = poses
        return data


if __name__ == "__main__":
    dataLoader = DataLoader()
    dataLoader.get_blender_data("chair")
