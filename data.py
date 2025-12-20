import numpy as np
import os
import json


class DataLoader:
    def __init__(self):
        self.base_dir = "./data/nerf_synthetic/"

    #
    def get_blender_data(self, type="train"):
        metas = []
        for image_type in os.listdir(
            self.base_dir
        ):  # go through each image type in dataset
            base_image_path = os.path.join(self.base_dir, image_type)
            if not os.path.isfile(base_image_path):  # avoid the README
                for file in os.listdir(base_image_path):
                    if file == type:
                        print("a")

        return

    def get_blender_training():
        return


if __name__ == "__main__":
    dataLoader = DataLoader()
    dataLoader.get_blender_data()
