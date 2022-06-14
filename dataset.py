import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

"""
From Pytorch doc:
All pre-trained models expect input images normalized in the same way, 
i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W 
are expected to be at least 224. The images have to be loaded in to a range of [0, 1] 
and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
"""


def select_transformer(pre):
    # Transforms
    if pre:
        transformer = transforms.Compose([
            transforms.Resize(256),  # Resize to 256 x 256
            transforms.CenterCrop(224),  # Crop from center to convert to 224 x 224
            transforms.ToTensor(),  # 0-255 to 0-1, numpy to tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])  # Mean and std of pretained set
        ])
    else:
        transformer = transforms.Compose([
            transforms.Resize(256),  # Resize to 256 x 256
            transforms.CenterCrop(224),  # Crop from center to convert to 224 x 224
            transforms.ToTensor(),  # 0-255 to 0-1, numpy to tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5],  # 0-1 to [-1,1] , formula (x-mean)/std
                                 std=[0.5, 0.5, 0.5])
        ])

    return transformer


def load_train(train_path, pretrained, batch_size):
    transformer = select_transformer(pretrained)
    train_loader = DataLoader(
        torchvision.datasets.ImageFolder(train_path, transform=transformer),
        batch_size=batch_size, shuffle=True
    )
    return train_loader


def load_test(test_path, pretrained, batch_size):
    transformer = select_transformer(pretrained)
    train_loader = DataLoader(
        torchvision.datasets.ImageFolder(test_path, transform=transformer),
        batch_size=batch_size, shuffle=True
    )
    return train_loader
