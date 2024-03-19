"""Script with plotting ancillar functions."""

import math
import yaml
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler as schedulers
from matplotlib import pyplot as plt


# Use an LLM for the docs comment asking to
#   write a documentation comment for the following function considering the geranel idea, inputs and outputs using the style of the first one
# then copy the answer in the script.

def plot_multiple_images(images, titles=None, rows=1, cols=None):
    """
    Plots multiple images in a grid layout.

    Args:
        images: A list of image arrays.
        titles: A list of titles for each image (optional).
        rows: Number of rows in the layout (default: 1).
        cols: Number of columns in the layout (default: None, calculated based on rows and total images).
    """

    # Check if the number of images matches the length of titles (if provided)
    if titles and len(images) != len(titles):
        raise ValueError("Number of images and titles must match.")

    # Calculate number of columns if not provided
    if not cols:
        cols = int(math.ceil(len(images) / rows))

    # Initialize figure and subplots
    _, axes = plt.subplots(rows, cols, figsize=(max(len(images)*5, 5),rows*5))  # Adjust figure size as needed

    # Loop through images and titles
    for i, (ax, image) in enumerate(zip(axes.flat, images)):

        image = np.array(image)
        image = image/image.max() if image.max() >1 else image
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        # Add title if provided
        if titles:
            ax.set_title(titles[i])

    # Adjust layout
    #fig.suptitle(f"{len(images)} Images", fontsize=12)
    plt.tight_layout()
    plt.show()
