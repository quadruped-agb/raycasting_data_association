"""
Makes the pixel<->3D-point correspondence VISUALLY OBVIOUS, side by
side with the real photo -- more convincing than sparse dots.

For a given frame_id, this builds a new image, the SAME SIZE as the
original photo, that is entirely BLACK except at every pixel that
raycasting actually matched -- those pixels get their real color,
copied directly from the original photo at that exact (u, v).

Since matched pixels keep their real color and real position, the
result looks like a literal cutout/puzzle-piece of the original photo
(recognizable trunks, branches, edges), sitting in the exact position
they occupy in the real image. Put this side by side with the actual
photo: if raycasting/data association is correct, the colored shapes
in this reconstruction should visibly align with, and look like
fragments of, the real photo. If something's wrong, the colored
shapes will look displaced, scattered, or unrelated to the real
photo's actual content at that position.

    python3 tests/render_matched_reconstruction.py <frame_id> [--dir DIR] [--suffix SUFFIX]

  --dir DIR       directory to read frame_<id>_* files from
                   (default: output)
  --suffix SUFFIX suffix used in the saved filenames, e.g. "gputest"
                   for frame_<id>_pixel_to_point_gputest.npy /
                   frame_<id>_image_gputest.png (default: "" -- plain
                   frame_<id>_pixel_to_point.npy / frame_<id>_image.*)


Always writes its results into tests/ (frame_<id>_reconstruction.png,
frame_<id>_side_by_side.png), regardless of where the inputs came
from, so outputs from different variants don't get mixed up with the
main pipeline's own output/ folder.
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2

OUTPUT_DIR = "tests"


def find_frame_image(input_dir, fid, suffix):
    suffix_part = f"_{suffix}" if suffix else ""
    matches = glob.glob(os.path.join(input_dir, f"frame_{fid}_image{suffix_part}.*"))
    if not matches:
        raise FileNotFoundError(
            f"No saved image found for frame {fid} "
            f"(looked for {input_dir}/frame_{fid}_image{suffix_part}.*)."
        )
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frame_id", type=int)
    parser.add_argument("--dir", default="output", help="directory to read inputs from (default: output)")
    parser.add_argument("--suffix", default="", help='e.g. "gputest" (default: none, plain filenames)')
    args = parser.parse_args()

    fid = args.frame_id
    input_dir = args.dir
    suffix = args.suffix
    suffix_part = f"_{suffix}" if suffix else ""

    pixel_to_point_path = os.path.join(input_dir, f"frame_{fid}_pixel_to_point{suffix_part}.npy")
    if not os.path.exists(pixel_to_point_path):
        raise FileNotFoundError(f"{pixel_to_point_path} not found.")

    image_path = find_frame_image(input_dir, fid, suffix)
    print(f"Frame {fid}")
    print(f"  pixel_to_point : {pixel_to_point_path}")
    print(f"  image          : {image_path}")

    pixel_to_point = np.load(pixel_to_point_path, allow_pickle=True).item()
    if not pixel_to_point:
        print("This frame has no matched pixels -- nothing to reconstruct.")
        sys.exit(0)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    reconstruction = np.zeros_like(image)
    for (u, v) in pixel_to_point.keys():
        reconstruction[v, u] = image[v, u]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # tag output filenames with the suffix too, so results from
    # different variants (plain / gputest / etc.) don't overwrite
    # each other in tests/
    tag = f"_{suffix}" if suffix else ""

    recon_path = os.path.join(OUTPUT_DIR, f"frame_{fid}_reconstruction{tag}.png")
    cv2.imwrite(recon_path, reconstruction)
    print(f"\nSaved reconstruction: {recon_path}")

    separator = np.full((image.shape[0], 4, 3), 255, dtype=np.uint8)
    side_by_side = np.hstack([image, separator, reconstruction])
    side_path = os.path.join(OUTPUT_DIR, f"frame_{fid}_side_by_side{tag}.png")
    cv2.imwrite(side_path, side_by_side)
    print(f"Saved side-by-side (original | reconstruction): {side_path}")

    matched_count = len(pixel_to_point)
    total_pixels = image.shape[0] * image.shape[1]
    print(f"\nMatched {matched_count} / {total_pixels} pixels "
          f"({100 * matched_count / total_pixels:.1f}% coverage)")
    print("\nOpen the side-by-side image: the colored shapes on the right")
    print("should look like recognizable fragments of the photo on the")
    print("left, in the same positions (trunks, branches, edges lining")
    print("up). If they look displaced or unrelated, that's a real")
    print("correspondence problem -- not just a viewing-angle illusion.")


if __name__ == "__main__":
    main()