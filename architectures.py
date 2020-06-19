import torch
# from torchvision.models.resnet import resnet50
import torch.backends.cudnn as cudnn
from archs.cifar_resnet import resnet as resnet_cifar
from archs.cifar_resnet import lenet300
from archs.cifar_resnet import lenet5, lenet_5_caffe
from archs.cifar_resnet import fcn, wide_resnet
from archs.cifar_resnet import vgg19, resnet32, resnet50
from torch.nn.functional import interpolate


ARCHITECTURES = ["resnet50", "lenet300", "lenet5", "vgg19", "resnet32", "vgg16", "lenet_5_caffe"]
def get_architecture(arch: str, dataset: str, device) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if arch == "resnet50" and dataset == "imagenet":
        model = torch.nn.DataParallel(resnet50(pretrained=False)).to(device)
        cudnn.benchmark = True
    elif arch == "resnet50" and dataset == "tiny_imagenet":
        model = resnet50(num_classes=200).to(device)
    elif arch == "resnet32" and dataset == "tiny_imagenet":
        model = resnet32(num_classes=200).to(device)
    elif arch == "resnet32" and dataset == "cifar100":
        model = resnet32(num_classes=100).to(device)
    elif arch == "resnet32":
        model = resnet32(num_classes=10).to(device)
    elif arch == "vgg19" and dataset == "tiny_imagenet":
        model = vgg19(num_classes=200).to(device)
    elif arch == "lenet300":
        model = lenet300(num_classes=10).to(device)
    elif arch == "lenet5":
        model = lenet5(num_classes=10).to(device)
    elif arch == "lenet_5_caffe":
        model = lenet_5_caffe().to(device)
    elif arch == "vgg19" and dataset == "cifar100":
        model = vgg19(num_classes=100).to(device)
    elif arch == "vgg19":
        model = vgg19(num_classes=10).to(device)
    return model