import fire
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.ndimage.morphology import binary_closing, binary_opening


def dir(input_directory, output_directory):
  """Crop away static pixels from all video files in a directory

  Examples

  sonocrop dir input-dataset output-dataset


  Args:
      input_directory: Directory of video files
      output_directory: Directory of output, must be empty
  """

  import os
  import rich
  from pathlib import Path
  if not os.path.exists(output_directory):
    os.makedirs(output_directory)

  paths = Path(input_directory).glob('**/*.mp4')
  for path in paths:
    output_file = Path(output_directory)/path.name
    crop(path, output_file)



def crop(input_file, output_file, thresh=0.1):
  """Crop away static pixels from an ultrasound

  This function isolates the moving video in the center of an ultrasound clip and
  removes static borders that often contains patient information.

  Examples

  sonocrop crop in.mp4 out.mp4 --thresh=0.05


  Args:
      input_file: Path to input video (must be mp4, mov, or avi)
      output_file: File path for video output
      thresh (float, optional): Defaults to 0.1
  """

  import numpy as np
  import cv2
  from pathlib import Path
  from sonocrop import vid

  import rich
  from rich.progress import Progress

  v, fps, f, height, width = vid.loadvideo(input_file)

  rich.print(f'Auto cropping: [underline]{input_file}[/underline]')
  rich.print(f'  Frames: {f}')
  rich.print(f'  FPS: {fps}')
  rich.print(f'  Width x height: {width} x {height}')
  rich.print(f'  Thresh: {thresh}')

  # Count unique pixels
  with Progress() as progress:
    task = progress.add_task("[green] Finding static video pixels...", total=height)
    u = np.zeros((height, width), np.uint8)
    for i in range(height):
        u[i] = np.apply_along_axis(vid.countUniquePixels, 0, v[:,i,:])
        progress.update(task, advance=1)

  u_avg = u/f

  rich.print(' Finding edges')

  maxW = np.max(u_avg, axis=0)
  left,right = vid.findEdges(maxW, thresh=thresh)
  maxH = np.max(u_avg, axis=1)
  top,bottom = vid.findEdges(maxH, thresh=thresh)

  rich.print(f'  Top: {top}')
  rich.print(f'  Bottom: {bottom}')
  rich.print(f'  Left: {left}')
  rich.print(f'  Right: {right}')

  cropped = v[:,top:bottom,left:right]

  rich.print(f' Saving to file: "{output_file}"')
  vid.savevideo(output_file, cropped, fps)

  rich.print(' DONE')


def edges(input_file, thresh=0.1):
  """Extracts the edges around an ultrasound

  Returns the distance in pixels in the form:
  left,right,top,bottom

  Args:
      input_file: Path to input video (must be mp4, mov, or avi)
      thresh (float, optional): Defaults to 0.1
  """

  import numpy as np
  import cv2
  from pathlib import Path
  from sonocrop import vid
  v, fps, f, height, width = vid.loadvideo(input_file)

  u = vid.countUniquePixelsAlongFrames(v)
  u_avg = u/f

  maxW = np.max(u_avg, axis=0)
  left,right = vid.findEdges(maxW, thresh=thresh)
  maxH = np.max(u_avg, axis=1)
  top,bottom = vid.findEdges(maxH, thresh=thresh)

  return (f'{left},{right},{top},{bottom}')

def keep_largest_component(binary_image):
    # Label connected components in the binary image
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    # Find the largest component (excluding the background)
    largest_label = 1
    largest_area = stats[1, cv2.CC_STAT_AREA]
    for i in range(2, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_label = i

    # Create a new binary image containing only the largest component
    largest_component = np.zeros_like(binary_image)
    largest_component[labels == largest_label] = 255

    return largest_component
  
def sync_halves(binary_image):
    height, width = binary_image.shape

    # Split the image in half along the width
    left_half = binary_image[:, :width // 2]
    right_half = binary_image[:, width // 2:]

    # Flip the right half horizontally to align it with the left half
    right_half_flipped = np.fliplr(right_half)

    # Set a pixel to 1 in the left half if the corresponding symmetric pixel is 1 in the right half
    left_half[np.where(right_half_flipped == 255)] = 255

    # Flip the left half horizontally to align it with the right half
    left_half_flipped = np.fliplr(left_half)

    # Set a pixel to 1 in the right half if the corresponding symmetric pixel is 1 in the left half
    right_half[np.where(left_half_flipped == 255)] = 255

    # Combine the synchronized halves back into a single image
    synced_image = np.concatenate((left_half, right_half), axis=1)

    return synced_image

def crop_single_object(bool_image):
  # Find the non-zero elements' indices (i.e., the object's coordinates)
  y_coords, x_coords = np.nonzero(bool_image)

  # Determine the minimum and maximum x and y coordinates of the object
  xmin, xmax = np.min(x_coords), np.max(x_coords)
  ymin, ymax = np.min(y_coords), np.max(y_coords)

  # Crop the image using the calculated coordinates
  cropped_image = bool_image[ymin:ymax+1, xmin:xmax+1]

  return cropped_image, ymin, ymax+1, xmin, xmax+1,


def mask(input_file, output_file, thresh=0.05):
  """Blackout static pixels in an ultrasound

  Examples

  sonocrop mask in.mp4 out.mp4 --thresh=0.05

  Args:
      input_file: Path to input video (must be mp4, mov, or avi)
      output_file: File path for video output
      thresh (float, optional): Defaults to 0.05
  """

  
  from pathlib import Path
  from sonocrop import vid

  import rich
  from rich.progress import Progress

  v, fps, f, height, width = vid.loadvideo(input_file)

  rich.print(f'Mask video: [underline]{input_file}[/underline]')
  rich.print(f'  Frames: {f}')
  rich.print(f'  FPS: {fps}')
  rich.print(f'  Width x height: {width} x {height}')
  rich.print(f'  Thresh: {thresh}')

  # Count unique pixels
  with Progress() as progress:
    task = progress.add_task("[green] Finding static video pixels...", total=height)
    u = np.zeros((height, width), np.uint8)
    for i in range(height):
        u[i] = np.apply_along_axis(vid.countUniquePixels, 0, v[:,i,:])
        progress.update(task, advance=1)

  u_avg = u/f
  mask = u_avg > thresh
  
  mask_img = mask.astype(np.uint8)

  
  cv2.imshow('mask_img',mask_img*255)
  cv2.waitKey(0)
  
  mask_largest_img = keep_largest_component(mask_img)
  
  
  cv2.imshow('mask_largest_img',mask_largest_img)
  cv2.waitKey(0)
  
  mask_mirrored_largest_img = sync_halves(np.copy(mask_largest_img))
  
  cv2.imshow('mask_mirrored_largest_img',mask_mirrored_largest_img)
  cv2.waitKey(0)
  
  boolean_mask = binary_fill_holes((mask_mirrored_largest_img/255).astype(bool))
  
  cv2.imshow('boolean_mask',boolean_mask.astype(np.uint8)*255)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
  # boolean_mask = binary_opening(boolean_mask, structure=np.ones((2,2)))
  
  # cv2.imshow('closed boolean_mask',boolean_mask.astype(np.uint8)*255)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
  
  # Find contours in the binary image
  binary_image = boolean_mask.astype(np.uint8)*255
  contours = None
  if cv2.__version__.startswith('3'):
      _, contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  else:
      contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      
  # Find the convex hull of the largest contour
  convex_hull = cv2.convexHull(contours[0])

  # Create a blank image with the same shape as the original binary image
  convex_hull_image = np.zeros_like(boolean_mask, dtype=np.uint8)

  # Draw the convex hull on the blank image
  cv2.drawContours(convex_hull_image, [convex_hull], -1, 255, thickness=cv2.FILLED)

  # Convert the image back to boolean
  convex_hull_image = convex_hull_image.astype(bool)
  
  cv2.imshow('convex_hull_image',convex_hull_image.astype(np.uint8)*255)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
  cropped_image, ymin, ymax, xmin, xmax = crop_single_object(np.copy(convex_hull_image))
  cropped_boolean_mask, _, _, _, _ = crop_single_object(np.copy(boolean_mask))
  
  
  # Save the binary image to disk
  print(f"{output_file[:-4]}_ch_mask{output_file[-4:]}")
  cv2.imwrite(f"{output_file[:-4]}_uncropped_ch_mask.png", (convex_hull_image * 255).astype(np.uint8))
  cv2.imwrite(f"{output_file[:-4]}_ch_mask.png", (cropped_image * 255).astype(np.uint8))
  cv2.imwrite(f"{output_file[:-4]}_cropped_boolean_mask.png", (cropped_boolean_mask * 255).astype(np.uint8))
  
  cv2.imshow('cropped_image', (cropped_image * 255).astype(np.uint8))
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  y = vid.applyMask(v, boolean_mask)
  y_cropped = y[:, ymin:ymax, xmin:xmax]
  rich.print(f' Saving to file: "{output_file}"')
  vid.savevideo(output_file, y_cropped, fps)
  
  import mmcv
  
  video = mmcv.VideoReader(output_file)
  video.cvt2frames(f"{output_file[:-4]}_frames")
  
  return ('Done')


def main():
  fire.Fire()

if __name__ == '__main__':
  main()
