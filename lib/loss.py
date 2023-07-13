"""
Implements loss functions. Some of the loss functions also have an accompanying prediction head layer implementation in the model.py module.
E.g., :py:meth:`megaage_loss` is designed to be paired with :class:`lib.model.MegaAgeLayer`.

The 'pairing' of different heads and losses is done in :class:`.Model`.

Functions:
    - :py:meth:`mean_variance_loss`
    - :py:meth:`unimodal_concentrated_loss`
    - :py:meth:`unimodal_concentrated_loss_not_official`
    - :py:meth:`dldl_loss`
    - :py:meth:`dldl_v2_loss`
    - :py:meth:`soft_labels_loss`
    - :py:meth:`orcnn_loss`
    - :py:meth:`coral_loss`
    - :py:meth:`megaage_loss`
    - :py:meth:`mae_loss`
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def mean_variance_loss(logits: torch.tensor, labels: torch.tensor, lambda_1=0.2, lambda_2=0.05):
    """
    Computes the Mean-Variance Loss as defined in Mean-Variance Loss for Deep Age Estimation from a Face.

    Args:
        logits (torch.tensor): Size(batch_size, nr_classes])
        labels (torch.tensor): Size([batch_size])
        lambda_1 (float, optional): Weight of the mean loss. Defaults to 0.2.
        lambda_2 (float, optional): Weight of the variance loss. Defaults to 0.05.

    """
    batch_size = logits.shape[0]
    nr_classes = logits.shape[1]

    probas = F.softmax(logits, dim=1)

    class_labels = torch.arange(0, nr_classes)
    class_labels = torch.broadcast_to(
        class_labels, probas.shape).to(labels.device)

    means = torch.sum(probas*class_labels, dim=1).to(labels.device)
    broadcast_means = torch.broadcast_to(
        means[:, None], probas.shape).to(labels.device)
    variances = torch.sum((broadcast_means - class_labels)
                          ** 2*probas, dim=1).to(labels.device)

    ce_loss = F.cross_entropy(logits, labels)
    mean_loss = torch.mean((means-labels)**2)/2.
    variance_loss = torch.mean(variances)

    loss = ce_loss + lambda_1*mean_loss + lambda_2*variance_loss
    return loss


def unimodal_concentrated_loss(logits: torch.tensor, labels: torch.tensor):
    """
    For the paper, we obtained the official implementation from the authors of Unimodal-Concentrated Loss. However, we were asked not to make it public. Therefore, the loss function has been removed from the release version of the code.
    """
    raise NotImplementedError(
        "For the paper, we obtained the official implementation from the authors of Unimodal-Concentrated Loss. However, we were asked not to make it public. Therefore, the loss function has been removed from the release version of the code.")


def unimodal_concentrated_loss_not_official(logits: torch.tensor, labels: torch.tensor, lambda_=1000):
    """
    Computes the Unimodal-Concentrated Loss as defined in Unimodal-Concentrated Loss: Fully Adaptive Label Distribution Learning for Ordinal Regression.

    Args:
        logits (torch.tensor): Size(batch_size, nr_classes])
        labels (torch.tensor): Size([batch_size])
        lambda_ (float, optional): Weight of the unimodal loss. Defaults to 1000.

    """
    batch_size = logits.shape[0]
    nr_classes = logits.shape[1]
    eps = 1e-10

    probas = F.softmax(logits, dim=1)
    diffs = torch.diff(probas, dim=1).to(labels.device)

    class_labels = torch.arange(0, nr_classes)
    class_labels = torch.broadcast_to(
        class_labels, probas.shape).to(labels.device)

    broadcast_labels = torch.broadcast_to(
        labels[:, None], probas.shape).to(labels.device)

    select = []
    for label in labels:
        select.append(torch.tensor(
            [i for i in range(label)] + [i for i in range(label+1, nr_classes)]))
    select = torch.stack(select).to(labels.device)

    sign = torch.gather(torch.sign(
        class_labels-broadcast_labels), 1, select).to(labels.device)

    signed_and_clipped_diffs = F.relu(sign*diffs).to(labels.device)
    unimodal_loss = torch.mean(
        torch.sum(signed_and_clipped_diffs, dim=1)).to(labels.device)

    means = torch.sum(probas*class_labels, dim=1).to(labels.device)
    broadcast_means = torch.broadcast_to(
        means[:, None], probas.shape).to(labels.device)
    variances = torch.sum(((broadcast_means - class_labels)**2)
                          * probas, dim=1).to(labels.device)

    assert torch.all(variances >= 0)

    concentrated_loss = torch.mean(
        torch.log(variances + eps) / 2. + ((means-labels)**2)/(2.*variances + eps))

    loss = concentrated_loss + lambda_ * unimodal_loss
    return loss


def dldl_loss(logits: torch.tensor, labels: torch.tensor, sigma=2):
    """
    Computes the loss as defined in Deep Label Distribution Learning With Label Ambiguity.

    Args:
        logits (torch.tensor): Size(batch_size, nr_classes])
        labels (torch.tensor): Size([batch_size])
        sigma (float, optional): Standard deviation of the Gaussian GT label distribution. Defaults to 2.

    """
    batch_size = logits.shape[0]
    nr_classes = logits.shape[1]

    probas = F.softmax(logits, dim=1)

    class_labels = torch.arange(0, nr_classes)
    class_labels = torch.broadcast_to(
        class_labels, probas.shape).to(labels.device)
    broadcast_labels = torch.broadcast_to(
        labels[:, None], probas.shape).to(labels.device)

    sigmas = torch.ones_like(labels).to(labels.device)*sigma
    broadcast_sigmas = torch.broadcast_to(
        sigmas[:, None], probas.shape).to(labels.device)
    label_distributions = torch.exp(-((class_labels - broadcast_labels)**2)/(
        2*broadcast_sigmas**2)) / (torch.sqrt(2*torch.pi*broadcast_sigmas))
    label_distributions = label_distributions / torch.broadcast_to(
        torch.sum(label_distributions, dim=1, keepdim=True), label_distributions.shape)

    loss = F.cross_entropy(logits, label_distributions)
    return loss


def dldl_v2_loss(logits: torch.tensor, labels: torch.tensor, sigma: float = 2, lambda_: float = 1):
    """
    Computes the loss as defined in Learning Expectation of Label Distribution for Facial Age and Attractiveness Estimation.

    Args:
        logits (torch.tensor): Size(batch_size, nr_classes])
        labels (torch.tensor): Size([batch_size])
        sigma (float, optional): Standard deviation of the Gaussian GT label distribution. Defaults to 2.
        lambda_ (float, optional): Weight of L1 loss. Defaults to 1.

    """
    batch_size = logits.shape[0]
    nr_classes = logits.shape[1]

    probas = F.softmax(logits, dim=1)
    diffs = torch.diff(probas, dim=1)

    class_labels = torch.arange(0, nr_classes)
    class_labels = torch.broadcast_to(
        class_labels, probas.shape).to(labels.device)
    broadcast_labels = torch.broadcast_to(
        labels[:, None], probas.shape).to(labels.device)

    sigmas = torch.ones_like(labels).to(labels.device)*sigma
    broadcast_sigmas = torch.broadcast_to(
        sigmas[:, None], probas.shape).to(labels.device)
    label_distributions = torch.exp(-((class_labels - broadcast_labels)**2)/(
        2*broadcast_sigmas**2)) / (torch.sqrt(2*torch.pi*broadcast_sigmas))
    label_distributions = label_distributions / torch.broadcast_to(
        torch.sum(label_distributions, dim=1, keepdim=True), label_distributions.shape)

    means = torch.sum(probas*class_labels, dim=1)

    loss = F.cross_entropy(logits, label_distributions) + \
        lambda_ * torch.mean(torch.abs(means-labels))
    return loss


def soft_labels_loss(logits: torch.tensor, labels: torch.tensor, distance_squared=False):
    """
    Computes the loss as defined in Soft Labels for Ordinal Regression.

    Args:
        logits (torch.tensor): Size(batch_size, nr_classes])
        labels (torch.tensor): Size([batch_size])
        distance_squared (bool, optional): If True, measures label distance as L2 instead of L1 for generating label distribution. Defaults to False.

    """
    batch_size = logits.shape[0]
    nr_classes = logits.shape[1]

    def abs_diff(a, b):
        return torch.abs(a-b)

    def squared_diff(a, b):
        return (a-b)**2

    def distance_measure(a, b):
        if distance_squared:
            return squared_diff(a, b)
        else:
            return abs_diff(a, b)

    probas = F.softmax(logits, dim=1)
    diffs = torch.diff(probas, dim=1)

    class_labels = torch.arange(0, nr_classes)
    class_labels = torch.broadcast_to(
        class_labels, probas.shape).to(labels.device)
    broadcast_labels = torch.broadcast_to(
        labels[:, None], probas.shape).to(labels.device)

    label_distributions = - \
        distance_measure(class_labels, broadcast_labels).type(logits.dtype)
    label_distributions = torch.softmax(label_distributions, dim=1)

    loss = F.cross_entropy(logits, label_distributions)
    return loss


def orcnn_loss(logits: torch.tensor, labels: torch.tensor, lambdas=None):
    """
    Computes the loss as defined in Ordinal Regression with Multiple Output CNN for Age Estimation.

    Args:
        logits (torch.tensor): Size(batch_size, nr_classes - 1])
        labels (torch.tensor): Size([batch_size])
        lambdas (torch.tensor): Weight of the binary classifiers.

    """
    batch_size = logits.shape[0]
    nr_classes = logits.shape[1] + 1

    # the authors claim that it is slightly better to set lambda_k = (sqrt(N_k)) / (sum sqrt(N_i))
    # where N_k is the number of samples of rank k
    if lambdas is None:
        lambdas = torch.ones([nr_classes - 1]).to(labels.device)

    probas = torch.sigmoid(logits)

    extended_labels = []
    for label in labels:
        if isinstance(label, torch.Tensor):
            int_label = label.item()
        else:
            int_label = label

        extended_label = [1]*int_label + [0] * (nr_classes - 1 - int_label)
        extended_label = torch.tensor(extended_label, dtype=probas.dtype)
        extended_labels.append(extended_label)

    extended_labels = torch.stack(extended_labels).to(labels.device)

    broadcast_lambdas = torch.broadcast_to(
        lambdas[None, :], extended_labels.shape).to(labels.device)

    loss = F.binary_cross_entropy(probas, extended_labels, reduction='none')

    # sum over binary classifications for each image, mean over batch
    loss = torch.mean(torch.sum(loss*broadcast_lambdas, dim=1), dim=0)

    return loss


def coral_loss(logits: torch.tensor, labels: torch.tensor):
    """
    Computes the CORAL loss as defined in Rank-consistent Ordinal Regression for Neural Networks.

    Args:
        logits (torch.tensor): Size(batch_size, nr_classes - 1])
        labels (torch.tensor): Size([batch_size])

    """
    batch_size = logits.shape[0]
    nr_classes = logits.shape[1] + 1
    extended_labels = []
    for label in labels:
        if isinstance(label, torch.Tensor):
            int_label = label.item()
        else:
            int_label = label

        extended_label = [1]*int_label + [0] * \
            (nr_classes - 1 - int_label)
        extended_label = torch.tensor(extended_label)
        extended_labels.append(extended_label)

    extended_labels = torch.stack(extended_labels).type(
        logits.dtype).to(labels.device)

    loss = (F.logsigmoid(logits)*extended_labels
            + (F.logsigmoid(logits) - logits)*(1-extended_labels))
    loss = (-torch.sum(loss, dim=1))
    loss = torch.mean(loss)
    return loss


def megaage_loss(logits: torch.tensor, labels: torch.tensor, truncation_distance: int = 3, sigma: float = 2.):
    """
    Computes the loss for a head combining extended binary classification and label distribution learning from Quantifying Facial Age by Posterior of Age Comparisons.

    Args:
        logits (torch.tensor): Size(batch_size, nr_classes + nr_classes - 1]) Contains [nr_classes] logits of output distribution and [nr_classes-1] binary probabilities for an extended binary classification task.
        labels (torch.tensor): Size([batch_size]) 
        truncation_distance (int, optional): Extended binary misclassification closer than the distance is ignored. Defaults to 3.
        sigma (float, optional): Sigma for Gaussian distribution of posterior. Defaults to 2..

    """
    batch_size = logits.shape[0]
    nr_classes = int((logits.shape[1] + 1) / 2)
    logits, binary_probas = logits[:, :nr_classes], logits[:, nr_classes:]
    probas = F.softmax(logits, dim=1)

    # "Hyperplane loss"
    truncations = []
    extended_labels = []
    for label in labels:
        if isinstance(label, torch.Tensor):
            int_label = label.item()
        else:
            int_label = label

        extended_label = [1]*int_label + [0] * (nr_classes - 1 - int_label)
        extended_label = torch.tensor(
            extended_label, dtype=binary_probas.dtype)

        # construct truncated cost
        truncation = torch.ones_like(extended_label)
        truncation[max(int_label-truncation_distance, 0)
                       :min(int_label+truncation_distance, nr_classes - 1)] = 0

        extended_labels.append(extended_label)
        truncations.append(truncation)

    extended_labels = torch.stack(extended_labels).to(labels.device)
    truncations = torch.stack(truncations).to(labels.device)

    hyper_loss = (binary_probas - extended_labels)**2
    hyper_loss = torch.mean(
        torch.sum(hyper_loss*truncations, dim=1), dim=0).to(labels.device)

    # Cross-entropy loss, where target distribution is a Gaussian
    class_labels = torch.arange(0, nr_classes)
    class_labels = torch.broadcast_to(
        class_labels, probas.shape).to(labels.device)
    broadcast_labels = torch.broadcast_to(
        labels[:, None], probas.shape).to(labels.device)

    sigmas = torch.ones_like(labels).to(labels.device)*sigma
    broadcast_sigmas = torch.broadcast_to(
        sigmas[:, None], probas.shape).to(labels.device)
    label_distributions = torch.exp(-((class_labels - broadcast_labels)**2)/(
        2*broadcast_sigmas**2)) / (torch.sqrt(2*torch.pi*broadcast_sigmas))
    label_distributions = label_distributions / torch.broadcast_to(
        torch.sum(label_distributions, dim=1, keepdim=True), label_distributions.shape)

    ce_loss = F.cross_entropy(logits, label_distributions)

    loss = hyper_loss + ce_loss
    return loss


def mae_loss(logits: torch.tensor, labels: torch.tensor):
    """
    Computes MAE loss for a regression network.

    Args:
        logits (torch.tensor): Size(batch_size, 1]) Regression outputs.
        labels (torch.tensor): Size([batch_size]) 

    """
    loss = torch.mean(torch.abs(logits - labels[:, None].float()))
    return loss
