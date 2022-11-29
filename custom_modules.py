"""
Dataset and modules.
"""

import pandas as pd
from PIL import Image
import torch as T
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, Normalize
from utils import one_hot_encode


class RAFDataset(Dataset):
    def __init__(self, csv_file, n_labels, img_size=224, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.n_labels = n_labels
        self.img_size = img_size
        if transform is None:
            transform = ToTensor()
        self.transform = torchvision.transforms.Compose([
            transform, Resize(self.img_size), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        image_path = self.dataframe.loc[idx, "Path"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label = self.dataframe.loc[idx, "Emotion"]
        label = one_hot_encode(label, self.n_labels)
        return image, label

def get_model(model_args):
    """
    Load a resnet18 pretrained on imagenet. Unfrozen.
    """
    if model_args['model_type'] == 'ResNet18':
        return ResNet18(model_args)
    else:
        raise NotImplementedError()

def ResNet18(model_args):
    T.hub.set_dir("./download")
    model = T.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=model_args['pretrained'])
    # fc outputs unnormalized logits
    model.fc = T.nn.Linear(in_features=512, out_features=model_args['n_labels'], bias=True)
    
    # Freeze setting
    if model_args['frozen']:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = True
    
    return model