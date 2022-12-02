"""
Dataset and modules.
"""

import numpy as np
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
        # compute label distribution
        labels = np.array(self.dataframe["Emotion"])
        self.label_dist = {i: np.sum(labels==i) for i in range(n_labels)}
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        image_path = self.dataframe.loc[idx, "Path"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label = self.dataframe.loc[idx, "Emotion"]
        label = one_hot_encode(label, self.n_labels)
        filename = self.dataframe.loc[idx, "Name"]
        attrs = np.array([self.dataframe.loc[idx, "Race"], self.dataframe.loc[idx, "Gender"], self.dataframe.loc[idx, "Age"]])
        return image, label, filename, attrs
    
    def label_distribution(self):
        return self.label_dist

def get_lookup_table(csv_file):
    """
    Create a file lookup table using csv of dataset files. Key is image name, value is dictionary of attributes.
    """
    dataframe = pd.read_csv(csv_file)
    lookup_table = {}
    for i, r in dataframe.iterrows():
        lookup_table[r["Name"]] = {
            "Path": r["Path"],
            "Emotion": r["Emotion"],
            "EmotionLabel": r["EmotionLabel"],
            "Race": r["Race"],
            "RaceLabel": r["RaceLabel"],
            "Gender": r["Gender"],
            "Age": r["Age"],
        }
    return lookup_table

def get_model(model_args):
    """
    Return a model based on model_args.
    Required keys:
        - model_type
        - pretrained
        - n_labels
        - frozen
    """
    if model_args['model_type'] == 'ResNet18':
        return ResNet18(model_args)
    elif model_args['model_type'] == 'ResNet50':
        return ResNet50(model_args)
    else:
        raise NotImplementedError()

def load_model(model_path, model_args):
    """
    Load a model from provided path.
    """
    if model_args['model_type'] == 'ResNet18':
        model = ResNet18(model_args)
        model.load_state_dict(T.load(model_path))
        return model
    elif model_args['model_type'] == 'ResNet50':
        model = ResNet50(model_args)
        model.load_state_dict(T.load(model_path))
        return model
    else:
        raise NotImplementedError()

def ResNet18(model_args):
    """
    Return a ResNet18.
    """
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

def ResNet50(model_args):
    """
    Return a ResNet50.
    """
    T.hub.set_dir("./download")
    model = T.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=model_args['pretrained'])
    # fc outputs unnormalized logits
    model.fc = T.nn.Linear(in_features=2048, out_features=model_args['n_labels'], bias=True)
    
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
