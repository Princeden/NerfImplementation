import argparse
from colmap import COLMAP
from model import train
from pathlib import Path
from graphics_utils import load_blender_data
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="NerfColmap Pipeline",
        description="Command Line interface to train and implement NeRFs (neural radiance fields) from any set of images utilizing the Colmap image reconstruction library",
    )
    parser.add_argument(
        "--use_colmap", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True, help="Input dataset directory"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output directory for final model and any desired training logs",
    )

    parser.add_argument(
        "--save_checkpoints", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--save_images", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    if args.use_colmap:
        colmap = COLMAP(args.input, args.output)
        print("Reconstructing Images for Camera Intrinsics")
        all_images, poses, render_poses, hwf = colmap.get_nerf_data()
    else:
        print("Loading data from JSON")
        all_images, poses, render_poses, hwf, i_split = load_blender_data(args.input)
    train(
        args.output,
        all_images,
        poses,
        render_poses,
        hwf,
        args.save_checkpoints,
        args.save_images,
    )
