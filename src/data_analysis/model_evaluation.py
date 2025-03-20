"""Methods used for evaluatiing neural network models."""
import os
import numpy as np
import torch

import matplotlib.pyplot as plt
from src.data_analysis.model_training import get_device_helper

CMAP = "BrBG"
VISUALS_DIR = "visualizations"


def gather_conv_layers(model: torch.nn.Module) -> list[torch.nn.Conv2d]:
    """
    Gather all the convolutional layers in the given model.

    :param model: The model to gather the convolutional layers from

    :return: A list of all the convolutional layers in the model
    """
    conv_layers = []
    for child in model.children():
        if isinstance(child, torch.nn.Sequential):
            for child_child in child.children():
                if isinstance(child_child, torch.nn.Conv2d):
                    conv_layers.append(child_child)
                else:
                    continue
        else:
            continue
    
    return conv_layers


def gather_feature_maps(
    conv_layers: list[torch.nn.Conv2d],
    input_image: torch.Tensor,
) -> list[np.ndarray]:
    """
    Gather the feature maps of the given input image for the given convolutional layers.

    :param conv_layers: The convolutional layers to use for feature extraction
    :param input_image: The image to use for feature extraction

    :return: A list of the feature maps for the given input image
    """
    feature_maps = []
    for layer in conv_layers:
        input_image = layer(input_image)
        feature_maps.append(input_image.squeeze(0).cpu().detach().numpy())

    return feature_maps


def plot_feature_maps(
    feature_maps: list[np.ndarray],
    savefig: bool = True,
    cmap: str = CMAP,
) -> None:
    """
    Plot the feature maps for the given convolutional layers.

    :param feature_maps: The feature maps to plot
    :param savefig: Whether to save the figure or not

    :return: None
    """
    for j, feature_map in enumerate(feature_maps):
        columns = 8
        rows = feature_map.shape[0] // 8
        fig, axs = plt.subplots(rows, columns, figsize=(columns + 4, rows + 4), dpi=100)
        for i, ax in enumerate(axs.flat):
            ax.imshow(feature_map[i], cmap=cmap)
            ax.axis('off')
            plt.suptitle(f'Feature Maps for Convolutional Layer {j}', fontsize=20)
            plt.tight_layout()
        if savefig:
            if not os.path.exists(VISUALS_DIR):
                os.mkdir(VISUALS_DIR)
            plt.savefig(f'visualizations/feature_maps_conv_layer_{j}.png')
    plt.show()


def feature_maps_viz_runner(
    model: torch.nn.Module,
    input_image: torch.Tensor,
    save_feature_maps: bool = True,
    cmap: str = CMAP,
) -> None:
    """
    Plot the feature maps of the given model and the provided input image.

    :param model: The model to use for feature extraction
    :param input_image: The image to use for feature extraction

    :return: None
    """
    device = get_device_helper()
    model.to(device)
    model.eval()
    conv_layers = gather_conv_layers(model)

    input_image = input_image.unsqueeze(0).unsqueeze(0).float()
    input_image = input_image.to(device)

    feature_maps = gather_feature_maps(conv_layers, input_image)
    plot_feature_maps(feature_maps, savefig=save_feature_maps, cmap=cmap)
