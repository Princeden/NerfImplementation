import torch
import torch.nn as nn
from data import DataLoader


if __name__ == "__main__":
    testing = True
    if testing:
        data_loader = DataLoader()
        blender_data = data_loader.get_blender_data()
        print("Loaded Data")
        print(blender_data.keys())
