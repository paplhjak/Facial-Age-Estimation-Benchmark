<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/paplhjak/Facial-Age-Estimation-Benchmark">
    <img src="logo_c.png" alt="Logo" width="300" height="300">
  </a>
  <h1 align="center">Implementing New Methods</h1>
</div>

# Introduction

In this repository, we focus solely on age estimation methods that modify the standard classification approach by changing the last few layers of the neural network or the loss function. Although this may appear restrictive, it is essential to note that a majority of the methods proposed in the field fall into this category. By comparing methods that modify only a small part of the network, we aim to ensure a fair evaluation, as the remaining setup can be kept identical.

Before implementing new methods, one should be familiar with ["Using the Repository"](using_the_repository.md).

# What should I implement?

To implement a new method, we need to define the following:

- The prediction head
- The loss function
- Method for converting the prediction head output to posterior over the classes

We will look at how to do so and how to insert it into our framework in the following.

## Prediction Layer

For some methods, the prediction head might be standard linear layer. Some methods, however, require special architecture of the layer. For instance, the prediction layer of [CORAL](https://www.sciencedirect.com/science/article/pii/S016786552030413X) requires that the weight vector of the layer is shared by all outputs, but different biases are used.

We implement the specialized prediction layers in `lib.model.py`.

For example, the definition of the [CORAL](https://www.sciencedirect.com/science/article/pii/S016786552030413X) layer is:

```python
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
```

The layer needs to inherit from `nn.Module` and only has to define the `__init__` and `forward` methods.

The `__init__` method receives two arguments, the dimensionality of the feature space (i.e., size of the layer input) and the number of classes (i.e., how many possible predictions should the layer support).

The `forward` method receives the feature representation (`torch.tensor`), typically of shape `[Batch, Nr Features]`.

For instance, the minimal working prediction head, which implements a standard fully-connected layer, can be defined as:

```python
class DummyLayer(nn.Module):
    def __init__(self, size_in, num_classes):
        super().__init__()
        self.size_in = size_in
        self.num_classes = num_classes
        self.layer = nn.Linear(self.size_in, self.num_classes, bias=True)

    def forward(self, x):
        return self.layer(x)
```

## Loss Function

Most of the methods for age estimation modify the standard approach by changing the loss function. For example, the [Mean-Variance loss](https://ieeexplore.ieee.org/document/8578652) keeps the standard Cross-Entropy loss, but also adds additional loss terms to it. Other methods require a completely custom loss function.

We implement the specialized loss functions in `lib.loss.py`.

For example, the definition of the [CORAL](https://www.sciencedirect.com/science/article/pii/S016786552030413X) loss function is:

```python
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
```

The loss function receives as input:

- `logits`: Outputs of the prediction head. The name `logits` is taken from the name of the standard linear layer output.
- `labels`: Groundtruth labels.

Output of the loss function should be equal to mean of the loss computed over individual samples of the mini-batch.

For instance, the minimal working loss function, which implements standard Cross-Entropy, can be defined as:

```python
def dummy_loss(logits: torch.tensor, labels: torch.tensor):
    batch_size = logits.shape[0]
    nr_classes = logits.shape[1]

    loss = F.cross_entropy(logits, labels, reduction='mean')
    return loss
```

## Posterior Computation

Our framework requires that all prediction heads implement a method for converting the layer output into a posterior of the possible classes. For instance, with standard linear layer, the posterior can be obtained as `softmax` of the `logits`, i.e.,

```python
torch.softmax(logits, 1)
```

Other methods might require a more nuanced computation. For methods which predict the age directly, such as with regression, we represent the posterior as a one-hot vector.

# Inserting the new implementation into the framework

When we are done implementing our prediction layer and a corresponding loss function, we can insert them into the framework.

To this end, we need to modify `lib.model.Model`. Specifically, we need to add our implementation to the following methods:
- `add_head(...)`
- `get_head_loss(...)`
- `get_head_posterior(...)`

We will demonstrate how using the `DummyLayer` and `dummy_loss(...)` showcased above.

First, we need to choose a name for our method. Let us use the name `my_new_method`. When someone wants to use our method, they will specify it in the [configuration](configuration_file.md) file as:

```
heads:
    - tag: "age"
      type: "my_new_method"
      ...
```

We therefore need to make sure that when the type `my_new_method` is specified, our implementation is used.

Second, in the `add_head(...)` method, we want to make sure that our `DummyLayer` is used. To this end, we will add the following `elif` clause into the switch in `add_head(...)`:

```python
elif self.head_types[tag] == 'my_new_method':
    self.heads.add_module(tag, DummyLayer(self.nr_features, self.head_nr_classes[tag]))
```

Third, in the `get_head_loss(...)` method, we want to make sure that our `dummy_loss` is used whenever the `DummyLayer` is used. To this end, we will add the following `elif` clause into the switch in `get_head_loss(...)`:

```python
elif self.head_types[tag] == 'my_new_method':
    return dummy_loss(logits, labels)
```

Lastly, we need to define the posterior. To this end, we will add the following `elif` clause into the switch in `get_head_posterior(...)`:

```python
elif self.head_types[tag] == 'my_new_method':
    return torch.softmax(logits, 1)
```

Congratulations! The `DummyLayer` and `dummy_loss` are now ready to be used by specified the head type `my_new_method` in the configuration file.
