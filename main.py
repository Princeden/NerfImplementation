import argparse
from COLMAP import COLMAP
from main.py import train
import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="NerfColmap Pipeline",
        description="Command Line interface to train and implement NeRFs (neural radiance fields) from any set of images utilizing the Colmap image reconstruction library",
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True, help="Input dataset directory"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output directory for checkpoints and logs",
    )
    args = parser.parse_args()
    colmap = COLMAP(parser.input, parser.output)
    print("Reconstructing Images for Camera Intrinsics")
    all_images, poses, render_poses, hwf = colmap.get_nerf_data()
    train(parser.input, parser.output, all_images, render_poses, hwf)
