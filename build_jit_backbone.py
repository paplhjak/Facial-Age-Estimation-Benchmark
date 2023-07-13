"""
This script wraps a trained model's backbone in a JIT script format.
It is required to specify the *configuration file* and the *checkpoint file*.

Creating a JIT backbone is useful in multitude of cases:
1) Wrapping a pre-trained model, e.g., pre-trained on IMDB-WIKI, and using the model as a backbone for multiple experiments.
2) Wrapping a "fancy" model without requiring to import it's code in order to use it for age estimation. This script, however, is not designed for this use case.

"""

import os
import yaml
import argparse
from lib.data_loaders import MyYamlLoader
from lib.model import initialize_model
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TTF


class GenericBackbone(nn.Module):
    def __init__(self, model):
        super(GenericBackbone, self).__init__()
        backbone = model.model

        # Store the callable backbone in `self.model`
        self.model = backbone

    def forward(self, x):
        # Define the forward pass of the backbone.
        # Sometimes, it is useful to include interpolation to a desired size, as some models require a specific input shape.
        if TTF.get_image_size(x) != (256, 256):
            x = torch.nn.functional.interpolate(x,
                                                size=(256, 256),
                                                mode='bilinear',
                                                antialias=True).float()
        # Extract the features
        x = self.model(x)
        return x


if __name__ == "__main__":
    # Get input arguments
    parser = argparse.ArgumentParser(
        description="Compiles a trained model into a JIT scripted model.")
    parser.add_argument(
        "config", type=str, help='Path to configuration file used to train the model.')
    parser.add_argument("checkpoint_file", type=str,
                        help='Path to checkpoint file with trained weights.')
    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        sys.exit(f"Config file {args.config} does not exist.")
    with open(args.config, 'r') as stream:
        config = yaml.load(stream, Loader=MyYamlLoader)

    # Build the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = initialize_model(config)
    model = model.to(device)

    assert os.path.exists(args.checkpoint_file)

    # Load best weights
    checkpoint = torch.load(args.checkpoint_file)
    model.load_state_dict(checkpoint['best_model_wts'])

    # Export to TorchScript
    model_scripted = torch.jit.script(GenericBackbone(model))

    # Save the scripted backbone
    torch.jit.save(model_scripted, "jit-backbone.pt")
