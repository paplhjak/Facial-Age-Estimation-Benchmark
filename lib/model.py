"""
Implements the overall model structure in :class:`Model`. 
Also implements :py:meth:`initialize_model` used to initialize the model.

Additionally, :class:`ExtendedBackbone` implements a wrapper around a feature extraction backbone, extending it with a simple multi-layer-perceptron.

Also defines :class:`MegaAgeLayer` and :class:`CoralLayer`, special prediction head implementations used in conjunction with :py:meth:`lib.loss.megaage_loss` and :py:meth:`lib.loss.coral_loss`.

Classes:
    - :class:`Model`
    - :class:`ExtendedBackbone`
    - :class:`MegaAgeLayer`
    - :class:`CoralLayer`

Functions:
    - :py:meth:`initialize_model`
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, efficientnet_b0, efficientnet_b2, efficientnet_b4, efficientnet_b6, vit_b_16, vgg16_bn
from torchvision.models import ResNet50_Weights
from lib.loss import *
from typing import Dict


class Model(nn.Module):
    """
    Base model composed of a backbone feature extractor and prediction heads.
    """

    def __init__(self, backbone, nr_features, heads):
        """
        Args:
            backbone (nn.Module): PyTorch model which receives a Tensor and returns a feature representation, e.g., ResNet-50 without the last layer.
            nr_features (int): Number of features extracted by the backbone for each image.
            heads (dict): Dictionary defining the prediction heads, parsed from the configuration file.
                A simple heads dictionary might look as `{"tag": "age", "attribute": "age", "labels": [0, 1, ..., 101], "weight": 1, "metric": ["mae", "cs5"]}`
        """
        super(Model, self).__init__()

        self.nr_features = nr_features
        self.model = backbone

        self.heads = nn.ModuleDict()
        self.head_types = torch.jit.annotate(Dict[str, str], {
                                             head['tag']: head['type'] if 'type' in head else 'classification' for head in heads})

        self.head_nr_classes = torch.jit.annotate(Dict[str, int], {
            head['tag']: len(head['labels']) for head in heads})

        for head in heads:
            self.add_head(head['tag'])

    def add_head(self, tag: str):
        """
        Method which adds a prediction head, e.g., standard classification layer or some other layer defined specifically for age estimation.

        Within the configuration file, the prediction head needs to define multiple parameters, among which are:
            - tag: Unique identifier of the prediction head, e.g., 'age_prediction_head'.
            - type: The desired method to be implemented by the prediction head, e.g., 'classification' or 'mean_variance' or others.

        This functions constructs the prediction head, identified by it's unique tag and adds it a `nn.ModuleDict` containing all of the prediction heads.

        Args:
            tag (str): Unique identifier of the prediction head, i.e., name of the prediction head as specified in the configuration file.

        Raises:
            NotImplementedError: If type of the specified head is not implemented. 
        """
        if (self.head_types[tag] == 'classification') or \
            (self.head_types[tag] == 'dldl') or \
            (self.head_types[tag] == 'dldl_v2') or \
            (self.head_types[tag] == 'unimodal_concentrated') or \
            (self.head_types[tag] == 'soft_labels') or \
                (self.head_types[tag] == 'mean_variance'):
            self.heads.add_module(tag, nn.Linear(
                self.nr_features, self.head_nr_classes[tag]))

        elif self.head_types[tag] == 'regression':
            self.heads.add_module(tag, nn.Linear(self.nr_features, 1))

        elif self.head_types[tag] == 'megaage':
            self.heads.add_module(tag, MegaAgeLayer(
                self.nr_features, self.head_nr_classes[tag]))

        elif (self.head_types[tag] == 'orcnn') or (self.head_types[tag] == 'extended_binary_classification'):
            self.heads.add_module(tag, nn.Linear(
                self.nr_features, self.head_nr_classes[tag] - 1))

        elif self.head_types[tag] == 'coral':
            self.heads.add_module(tag, CoralLayer(
                self.nr_features, self.head_nr_classes[tag]))

        else:
            raise NotImplementedError()

    def get_head_loss(self, logits, labels, tag):
        """
        Method which computes the loss for a particular prediction head.
        For a standard classification head, the loss is Cross-Entropy.
        For regression, the loss is the Mean-Absolute-Error.
        For other methods, e.g., CORAL, the loss is implemented in the :py:mod:`lib.loss` module.

        Within the configuration file, the prediction head needs to define multiple parameters, among which are:
            - tag: Unique identifier of the prediction head, e.g., 'age_prediction_head'.
            - type: The desired method to be implemented by the prediction head, e.g., 'classification' or 'mean_variance' or others.

        Args:
            logits (torch.tensor): Output of the prediction head.
            labels (torch.tensor): Ground truth labels.
            tag (str): Unique identifier of the prediction head.

        Raises:
            NotImplementedError: If type of the specified head is not implemented.
        """

        # default: classification with categorical cross-enropy
        if self.head_types[tag] == 'classification':
            return F.cross_entropy(logits, labels)

        elif self.head_types[tag] == 'regression':
            return mae_loss(logits, labels)

        elif self.head_types[tag] == 'mean_variance':
            return mean_variance_loss(logits, labels)

        elif self.head_types[tag] == 'unimodal_concentrated':
            return unimodal_concentrated_loss(logits, labels)

        elif self.head_types[tag] == 'dldl':
            return dldl_loss(logits, labels)

        elif self.head_types[tag] == 'dldl_v2':
            return dldl_v2_loss(logits, labels)

        elif self.head_types[tag] == 'soft_labels':
            return soft_labels_loss(logits, labels)

        elif self.head_types[tag] == 'megaage':
            return megaage_loss(logits, labels)

        elif self.head_types[tag] == 'orcnn':
            return orcnn_loss(logits, labels)

        elif (self.head_types[tag] == 'extended_binary_classification') or (self.head_types[tag] == 'coral'):
            return coral_loss(logits, labels)

        else:
            raise NotImplementedError()

    @torch.jit.export
    def get_head_posterior(self, logits, tag: str):
        """
        Method which computes the posterior probabiility of each class for a particular prediction head.
        All implemented methods need to define a posterior of the age, given the image.
        For a standard classification head, the posterior is computed by applying softmax to the logits.
        The same is true for some other methods, e.g., Mean-Variance loss or DLDL.

        Some methods do not predict a posterior, but instead predict the age itself.
        For such methods, e.g., regression, we construct the posterior probability as equal to 1 at the predicted age and 0 elsewhere.        

        We enforce that each method computes a posterior so that        For other methods, e.g., CORAL, the loss is implemented in the :py:mod:`lib.loss` module. it enables unified further evaluations.

        Args:
            logits (torch.tensor): Output of the prediction head.
            tag (str): Unique identifier of the prediction head. 
                Based on the tag, the prediction head type is determined.
                The method of computing the posterior is then decided based on the method/head type.

        Raises:
            NotImplementedError: If type of the specified head is not implemented.
        """
        if (self.head_types[tag] == 'classification') or \
            (self.head_types[tag] == 'dldl') or \
            (self.head_types[tag] == 'dldl_v2') or \
            (self.head_types[tag] == 'unimodal_concentrated') or \
            (self.head_types[tag] == 'soft_labels') or \
                (self.head_types[tag] == 'mean_variance'):
            return torch.softmax(logits, 1)

        elif self.head_types[tag] == 'regression':
            # encode the prediction as one-hot posterior
            logits = logits.flatten()
            predicted_labels = torch.minimum(torch.maximum(torch.round(logits), torch.zeros_like(
                logits)), torch.ones_like(logits)*(self.head_nr_classes[tag]-1)).long()
            posterior = F.one_hot(
                predicted_labels, num_classes=self.head_nr_classes[tag]).type(logits.dtype)
            return posterior

        elif self.head_types[tag] == 'megaage':
            return torch.softmax(logits[:, :self.head_nr_classes[tag]], 1)

        elif (self.head_types[tag] == 'orcnn') or (self.head_types[tag] == 'extended_binary_classification') or (self.head_types[tag] == 'coral'):
            # encode the prediction as one-hot posterior
            binary_probas = torch.sigmoid(logits)
            predicted_labels = torch.sum(binary_probas > 0.5, dim=1)
            posterior = F.one_hot(
                predicted_labels, num_classes=self.head_nr_classes[tag]).type(logits.dtype)
            return posterior

        elif self.head_types[tag] == 'coral_pseudo_posterior':
            binary_probas = torch.sigmoid(logits)
            # computes (1, p[0], p[1], ..., p[K]) - (p[0], p[1], ..., p[K], 0)
            # i.e., computes (1 - p[0], p[0]-p[1], ..., p[K] - 0)
            # for each sample in the mini batch
            A = torch.cat([torch.ones(binary_probas.shape[0], 1).type(
                binary_probas.dtype).to(binary_probas.device), binary_probas], dim=1)
            B = torch.cat([binary_probas, torch.zeros(binary_probas.shape[0], 1).type(
                binary_probas.dtype).to(binary_probas.device)], dim=1)
            # posterior, but not yet normalized
            pseudo_posterior = A - B
            # subtract minimal value
            min_vals, ixs = torch.min(pseudo_posterior, dim=1, keepdim=True)
            pseudo_posterior = pseudo_posterior - \
                torch.broadcast_to(min_vals, pseudo_posterior.shape)
            # normalize
            return pseudo_posterior / torch.broadcast_to(torch.sum(pseudo_posterior, dim=1, keepdim=True), pseudo_posterior.shape)

        else:
            raise NotImplementedError()

    def forward(self, x):
        """
        Forward pass of the model.
        Feature extraction and prediction head pass are explicitely separated to allow the heads to share a single feature extraction pass.

        Args:
            x (torch.tensor): Network input, typically a mini batch of images of the shape [Batch, Channels, Height, Width].
        """
        # Check for NaN in logits
        if torch.isnan(x).any():
            print("NaN detected in input!")
            print("Input:", x)
            raise ValueError("Inputs contain NaN values.")
        
        x = self.features(x)

        if torch.isnan(x).any():
            print("NaN detected in features!")
            print("Input:", x)
            raise ValueError("Features contain NaN values.")
        
        heads = self.logits(x)
        
        if torch.isnan(x).any():
            print("NaN detected in head!")
            print("Input:", heads)
            raise ValueError("Head contain NaN values.")
        
        return heads

    @torch.jit.export
    def features(self, x):
        """
        Extracts features of the input tensor. The function needs to be JIT scriptable to allow export.

        Args:
            x (torch.tensor): Network input, typically a mini batch of images of the shape [Batch, Channels, Height, Width].

        Returns:
            torch.tensor: Extracted features of shape [Batch, Nr Features]
        """
        return self.model(x).float()

    @torch.jit.export
    def logits(self, features):
        """
        Computes the prediction head outputs from extracted features.

        Args:
            features (torch.tensor): Extracted features of shape [Batch, Nr Features]

        Returns:
            Dictionary of prediction head outputs. 
        """
        return {name: head(features) for name, head in self.heads.items()}


def initialize_model(config: dict) -> nn.Module:
    """
    Initilizes the model.

    Args:
        config (dict): Configuration YAML parsed as a dictionary.
    Returns:
        model (nn.Module): Initialized model.
    """

    pretrained = False
    if 'use_pretrained' in config['model'].keys():
        pretrained = config['model']['use_pretrained']

    if config['model']['architecture'] == "jit":
        backbone = torch.jit.load(config['model']['path'])
        nr_features = config['model']['nr_features']

    elif config['model']['architecture'] == "resnet18":
        backbone = resnet18(pretrained=config['model']['use_pretrained'])
        nr_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

    elif config['model']['architecture'] == "resnet34":
        backbone = resnet34(pretrained=config['model']['use_pretrained'])
        nr_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

    elif config['model']['architecture'] == "resnet50":
        if pretrained:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            backbone = resnet50(pretrained=False)
        nr_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

    elif config['model']['architecture'] == "resnet101":
        backbone = resnet101(pretrained=config['model']['use_pretrained'])
        nr_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

    elif config['model']['architecture'] == "resnet152":
        backbone = resnet152(pretrained=config['model']['use_pretrained'])
        nr_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

    elif config['model']['architecture'] == "efficientnet_b0":
        backbone = efficientnet_b0(
            pretrained=config['model']['use_pretrained'])
        nr_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

    elif config['model']['architecture'] == "efficientnet_b2":
        backbone = efficientnet_b2(
            pretrained=config['model']['use_pretrained'])
        nr_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

    elif config['model']['architecture'] == "efficientnet_b4":
        backbone = efficientnet_b4(
            pretrained=config['model']['use_pretrained'])
        nr_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

    elif config['model']['architecture'] == "efficientnet_b6":
        backbone = efficientnet_b6(
            pretrained=config['model']['use_pretrained'])
        nr_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

    elif config['model']['architecture'] == "vit_b_16":
        if pretrained:
            backbone = vit_b_16(weights='DEFAULT')
        else:
            backbone = vit_b_16(pretrained=False)
        nr_features = 768
        backbone.heads = nn.Identity()

    elif config['model']['architecture'] == "vgg_16":
        if pretrained:
            backbone = vgg16_bn(weights='DEFAULT')
        else:
            backbone = vgg16_bn(pretrained=False)
        nr_features = 4096
        backbone.classifier[6] = nn.Identity()

    else:
        print("Unknown architecture.")
        exit()

    # Freeze / Unfreeze the backbone
    if 'backbone_trainable' in config['model'].keys():
        for ix, param in enumerate(backbone.parameters()):
            param.requires_grad = config['model']['backbone_trainable']

    # Add a MLP between the backbone and the prediction heads
    if 'intermediate_mlp_depth' in config['model'].keys():
        backbone = ExtendedBackbone(
            backbone=backbone, nr_features=nr_features, depth=config['model']['intermediate_mlp_depth'])

    return Model(backbone=backbone, nr_features=nr_features, heads=config['heads'])


class MegaAgeLayer(nn.Module):
    """
        Implements the prediction head from "Quantifying Facial Age by Posterior of Age Comparisons".

        The features are first processed by a linear layer and (nr_classes-1) logits corresponding to extended binary classification (EBC) task are predicted.
        Sigmoid of the binary logits is taken and the output is processed by a linear layer, outputing (nr_classes) logits corresponding to the class posterior.

        The binary probabilities and the output logits are concatenated and returned.
        This is done, because the loss is computed over both: 1) the EBC, 2) the output posterior.
    """

    def __init__(self, size_in, num_classes):
        """
        Args:
            size_in (int): Number of features extracted for each image.
            num_classes (int): Number of classes, i.e., label space cardinality.
        """
        super().__init__()
        self.size_in = size_in

        self.linear_1 = nn.Linear(self.size_in, num_classes - 1, bias=True)
        self.linear_2 = nn.Linear(num_classes - 1, num_classes, bias=True)

    def forward(self, x):
        """
        Forward pass of the layer.

        The features are first processed by a linear layer and (nr_classes-1) logits corresponding to extended binary classification (EBC) task are predicted.
        Sigmoid of the binary logits is taken and the output is processed by a linear layer, outputing (nr_classes) logits corresponding to the class posterior.

        The binary probabilities and the output logits are concatenated and returned.
        This is done, because the loss is computed over both: 1) the EBC, 2) the output posterior.
        """
        binary_logits = self.linear_1(x)
        binary_probas = torch.sigmoid(binary_logits)
        logits = self.linear_2(binary_probas)
        return torch.cat([logits, binary_probas], dim=1)


class CoralLayer(nn.Module):
    """
        Implements the prediction head from Rank consistent ordinal regression for neural networks with application to age estimation.
    """

    def __init__(self, size_in, num_classes, preinit_bias=True):
        """
        Args:
            size_in (int): Number of features extracted for each image.
            num_classes (int): Number of classes, i.e., label space cardinality.
            preinit_bias (bool, optional): If True, the biases are initialized to an ordered sequence. Defaults to True. 
        """
        super().__init__()
        self.size_in, self.size_out = size_in, 1
        self.out_features = num_classes - 1

        self.coral_weights = nn.Linear(self.size_in, 1, bias=False)
        if preinit_bias:
            self.coral_bias = nn.Parameter(
                torch.arange(num_classes - 1, 0, -1).float() / (num_classes-1))
        else:
            self.coral_bias = nn.Parameter(
                torch.zeros(num_classes-1).float())

    def forward(self, x):
        """
        Forward pass of the CORAL model. The weight vector of the logits is shared, but different biases are used.
        """
        return self.coral_weights(x) + self.coral_bias


class ExtendedBackbone(nn.Module):
    """
    Wrapper around a backbone model, adding a simple multi-layer-perceptron (MLP) at the end of the backbone.
    The width of the MLP is identical at all layers to the input size.

    Typical usage involves wrapping a pre-trained backbone with frozen weights.
    This allows us to preserve the learned representations, but also add expressiveness to the model.
    """

    def __init__(self, backbone, nr_features, depth):
        """
        Args:
            backbone (nn.Module): Backbone feature extractor.
            nr_features (int): Number of features extracted by the backbone for each image.
            depth (int): Number of MLP layers to build.
        """
        super().__init__()
        self.nr_features = nr_features
        self.depth = depth

        self.backbone = backbone

        layers = [nn.Linear(nr_features, nr_features),
                  nn.ReLU()]*depth

        self.MLP = nn.Sequential(*layers)

    def forward(self, x):
        """
        Calls forward pass of the backbone, then calls forward pass of the MLP.
        """
        return self.MLP(self.backbone(x).float())
