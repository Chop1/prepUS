import fire
import cv2
import numpy as np
import rich
from rich.progress import Progress
from pathlib import Path
from sonocrop import vid
from scipy.ndimage import binary_fill_holes

import mmcv
from typing import Tuple

from .utils import *


def removeLayout(input_file: str, output_file: str, thresh: float = 0.05) -> str:
    """
    Blackout static pixels in an ultrasound.

    Args:
        input_file (str): Path to input video (must be mp4, mov, or avi).
        output_file (str): File path for video output.
        thresh (float, optional): Threshold value for counting unique pixels. Defaults to 0.05.
    Returns:
        str: A string indicating the operation is done.
    """

    v, fps, f, height, width = vid.loadvideo(input_file)

    rich.print(f"video: [underline]{input_file}[/underline]")
    rich.print(f"  Frames: {f}")
    rich.print(f"  FPS: {fps}")
    rich.print(f"  Width x height: {width} x {height}")
    rich.print(f"  Thresh: {thresh}")

    # Count unique pixels
    with Progress() as progress:
        task = progress.add_task("[green] Finding static video pixels...", total=height)
        u = np.zeros((height, width), np.uint8)
        for i in range(height):
            u[i] = np.apply_along_axis(vid.countUniquePixels, 0, v[:, i, :])
            progress.update(task, advance=1)

    u_avg = u / f
    mask = u_avg > thresh

    mask_img = mask.astype(np.uint8)

    mask_largest_img = keep_largest_component(mask_img)

    mask_mirrored_largest_img = sync_halves(np.copy(mask_largest_img))

    boolean_mask = binary_fill_holes((mask_mirrored_largest_img / 255).astype(bool))

    cropped_boolean_mask, ymin, ymax, xmin, xmax = crop_single_object(np.copy(boolean_mask))

    # Save the binary image to disk
    cv2.imwrite(f"{output_file[:-4]}_cropped_boolean_mask.png", (cropped_boolean_mask * 255).astype(np.uint8))

    y = vid.applyMask(v, boolean_mask)
    y_cropped = y[:, ymin:ymax, xmin:xmax]
    rich.print(f"[green] Saving to file: [underline]{output_file}[/underline]")
    vid.savevideo(output_file, y_cropped, fps)

    return "Done"

def main():
    fire.Fire()


if __name__ == "__main__":
    main()