import torch.nn as nn
import torchvision.models as models


class ConvNet(nn.Module):
    def __init__(self, num_class):
        super(ConvNet, self).__init__()

        # Output size after convolution filter
        # ((w-f+2P)/s) + 1

        # Input shape = (256,3, 224, 224)
        # Batch Size, #Channels, width, height

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        # Shape= (256,12,224,224)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        # Shape= (256,12,224,224)
        self.relu1 = nn.ReLU()
        # Shape= (256,12,224,224)

        self.pool = nn.MaxPool2d(kernel_size=2)
        # Reduce the image size be factor 2
        # Shape= (256,12,112,112)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        # Shape= (256,20,112,112)
        self.relu2 = nn.ReLU()
        # Shape= (256,20,112,112)

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Shape= (256,32,112,112)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        # Shape= (256,32,112,112)
        self.relu3 = nn.ReLU()
        # Shape= (256,32,112,112)

        self.fc = nn.Linear(in_features=112 * 112 * 32, out_features=num_class)

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        output = output.view(-1, 32 * 112 * 112)

        output = self.fc(output)

        return output


# MobileNet V3
class MobileNet(nn.Module):
    def __init__(self, num_class, pretrained):
        super(MobileNet, self).__init__()
        self.model = models.mobilenet_v3_large(pretrained=pretrained)

        # in_features = 12800
        self.linear = nn.Linear(1280, num_class, bias=True)
        self.model.classifier[3] = self.linear

    def forward(self, input):
        return self.model.forward(input)


# ResNet18
class ResNet(nn.Module):
    def __init__(self, num_class, pretrained):
        super(ResNet, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)

        # in_features = 512
        self.linear = nn.Linear(512, num_class, bias=True)
        self.model.fc = self.linear

    def forward(self, input):
        return self.model.forward(input)


# VGG11
class VGG(nn.Module):
    def __init__(self, num_class, pretrained):
        super(VGG, self).__init__()
        self.model = models.vgg11(pretrained=pretrained)

        # in_features = 4096
        self.linear = nn.Linear(4096, num_class, bias=True)
        self.model.classifier[6] = self.linear

    def forward(self, input):
        return self.model.forward(input)


# AlexNet
class AlexNet(nn.Module):
    def __init__(self, num_class, pretrained):
        super(AlexNet, self).__init__()
        self.model = models.alexnet(pretrained=pretrained)

        # in_features = 4096
        self.linear = nn.Linear(4096, num_class, bias=True)
        self.model.classifier[6] = self.linear

    def forward(self, input):
        return self.model.forward(input)


def selectModel(modelName, n_class, pretrained):
    if modelName == 'resnet':
        model = ResNet(num_class=n_class, pretrained=pretrained)
    elif modelName == 'mobilenet':
        model = MobileNet(num_class=n_class, pretrained=pretrained)
    elif modelName == 'vgg':
        model = MobileNet(num_class=n_class, pretrained=pretrained)
    elif modelName == 'alexnet':
        model = MobileNet(num_class=n_class, pretrained=pretrained)
    else:
        model = ConvNet(num_class=n_class)

    return model
