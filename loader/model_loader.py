import collections
import settings
import torch
import torchvision

def get_vgg16(model_file):
    assert model_file is not None

    model = torchvision.models.vgg16()
    if settings.NUM_CLASSES != 1000:
        # Here, model.classifer[6] is the last FC
        model.classifier[6] = torch.nn.Linear(4096, settings.NUM_CLASSES)

    print("Setting up model.classifer to")
    print(model.classifier)

    print(f"Loading VGG16 from {model_file}")
    model.load_state_dict(torch.load(model_file))

    model.features = torch.nn.Sequential(
        collections.OrderedDict(
            zip(
                [
                    "conv1_1",
                    "relu1_1",
                    "conv1_2",
                    "relu1_2",
                    "pool1",
                    "conv2_1",
                    "relu2_1",
                    "conv2_2",
                    "relu2_2",
                    "pool2",
                    "conv3_1",
                    "relu3_1",
                    "conv3_2",
                    "relu3_2",
                    "conv3_3",
                    "relu3_3",
                    "pool3",
                    "conv4_1",
                    "relu4_1",
                    "conv4_2",
                    "relu4_2",
                    "conv4_3",
                    "relu4_3",
                    "pool4",
                    "conv5_1",
                    "relu5_1",
                    "conv5_2",
                    "relu5_2",
                    "conv5_3",
                    "relu5_3",
                    "pool5",
                ],
                model.features,
            )
        )
    )

    model.classifier = torch.nn.Sequential(
        collections.OrderedDict(
            zip(
                ["fc6", "relu6", "drop6", "fc7", "relu7", "drop7", "fc8a"],
                model.classifier,
            )
        )
    )

    # This is added by Pat;
    model.eval()

    return model

def loadmodel(hook_fn):
    if settings.MODEL == "vgg16":
        model = get_vgg16(model_file=settings.MODEL_FILE)
    elif settings.MODEL_FILE is None:
        model = torchvision.models.__dict__[settings.MODEL](pretrained=True)
    else:
        checkpoint = torch.load(settings.MODEL_FILE)
        if type(checkpoint).__name__ == 'OrderedDict' or type(checkpoint).__name__ == 'dict':
            model = torchvision.models.__dict__[settings.MODEL](num_classes=settings.NUM_CLASSES)
            if settings.MODEL_PARALLEL:
                state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
                    'state_dict'].items()}  # the data parallel layer will add 'module' before each layer name
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
        else:
            model = checkpoint

    for name in settings.FEATURE_NAMES:
        if settings.MODEL == "vgg16":
            print("Taking name", name)
            layer = model.features._modules.get(name)
        else:
            layer = model._modules.get(name)

        layer.register_forward_hook(hook_fn)
    if settings.GPU:
        model.cuda()
    model.eval()
    return model
