# Supported Backbones

The codebase supports multiple backbones for building deep learning models. Currently, it includes support for ResNet backbones, as well as the ability for users to implement their own custom backbones. Additionally, the codebase also supports using just-in-time (JIT) scripted backbones for ease of use and performance optimization.

## ResNet Backbones
The codebase provides support for ResNet backbones, including ResNet18, ResNet50, and others. Users can specify the desired ResNet architecture by editing the architecture field in the model configuration. For example, to use ResNet50, the configuration can be set as follows:

```
model:
  architecture: "resnet50"
  use_pretrained: True
  backbone_trainable: True
  input_size: [128, 128]
```

- If `use_pretrained` is set to `True`, the ResNet weights pretrained on ImageNet will be used.

- If `backbone_trainable` is set to `True`, the ResNet weights will be learned during training. If it is set to `False`, the weights will be frozen.

## JIT Scripted Backbones

The codebase also supports using JIT scripted backbones for the ease of use. JIT scripted backbones are serialized PyTorch models that can be loaded and used directly, without requiring additional dependencies, which is highly desirable.

To use a JIT scripted backbone, users can specify the `architecture` as "jit" and provide the file path to the JIT scripted backbone using the `path` field in the model configuration. Users must also specify the number of features extracted by the backbone using the `nr_features` field. For example:

```
model:
  architecture: "jit"
  path: "jit-scripted-backbone.pt"
  nr_features: 512
  backbone_trainable: False
  input_size: [128, 128]
```

- Note that the `backbone_trainable` field can be used to control whether the backbone should be trainable or not, same as for the ResNet backbones. 

Using a JIT scripted backbone is recommended for users who want to achieve the best performance with minimal effort in implementing custom backbones. The recommended approach is to use a large pre-trained visual transformer.

## MLP as intermediate processing of features before classifier heads
The `intermediate_mlp_depth` parameter in the model configuration allows the user to specify the depth of a simple MLP (Multi-Layer Perceptron) that is added to the output of the backbone model. The MLP has a constant number of features, and its depth is determined by the value provided for `intermediate_mlp_depth`.

### Example
Here is an example of how the intermediate_mlp_depth parameter can be used in the model configuration:
```
model:
  architecture: "jit"
  path: "FaRL-B16-224.pt"
  nr_features: 512
  intermediate_mlp_depth: 2
  backbone_trainable: False
  input_size: [128, 128]
```

### Typical Use
This functionality is commonly used when a pretrained backbone model is used, and the weights of the backbone model need to be frozen. It is useful when further processing of the feature space is required, for example, in ordinal regression with extended binary classification where classes are ordered in a linear direction. The MLP can help "straighten" the feature space to improve prediction accuracy.

## Handling Backbones with Specific Input Sizes
Please note that some backbones may require a specific input size, such as `[224,224]`. There are multiple ways to handle this requirement:

- **Option 1:** Use the input_size field in the model configuration to specify the desired input size, and then run data preparation scripts to obtain a version of the dataset with the desired size.

- **Option 2:** Include an interpolation operation directly in the JIT scripted backbone, as demonstrated in the script `build_backbone_train_eval.sh`, specifically in the file `build_jit_backbone.py`.

- **Option 3:** Define a resize operation in the data preprocessing configuration file of *Albumentations*.


*Note: Option 1 is the recommended approach as it is the default processing of a new experiment. However, if you want to compare a new backbone against previous experiments and therefore want to use already preprocessed data, Option 2 is recommended. Option 3 should generally not be used, but it is included for completeness.*

## Creating Custom JIT Backbone
A custom backbone can be implemented in the style of the provided `DummyBackbone`. The module is required to implement a `forward` function and store the underlying module in `self.model`.

```
class DummyBackbone(nn.Module):
    def __init__(self):
        super(DummyBackbone, self).__init__()
        # Build some backbone, e.g., ResNet-18 pretrained on ImageNet
        backbone = resnet18(pretrained=True)

        # Find out the size of the feature vector. If the model is trained with a head, we can remove unwanted layers by setting them to nn.Identity()
        nr_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        print(
            f"The backbone outputs {nr_features} features... Please set this in the configuration file.")

        # Store the callable backbone in `self.model`
        self.model = backbone

    def forward(self, x):
        # Define the forward pass of the backbone.
        # Sometimes, it is useful to include interpolation to a desired size, as some models require a specific input shape.
        x = torch.nn.functional.interpolate(x,
                                            size=(128, 128),
                                            mode='bilinear',
                                            antialias=True).float()
        # Extract the features
        x = self.model(x)
        return x
```

The backbone is then scripted as 
```
model_scripted = torch.jit.script(model)
torch.jit.save(model_scripted, "jit-scripted-backbone.pt")
```
