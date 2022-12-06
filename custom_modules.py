"""
Dataset and modules.
"""

import numpy as np
import pandas as pd
from PIL import Image
import torch as T
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, Normalize, RandomInvert
from utils import one_hot_encode

class RAFDatasetAugmented(Dataset):
    def __init__(self, csv_file, n_labels, img_size=224, transform=None, equalized_by="none", equalized_how="none"):
        self.dataframe = pd.read_csv(csv_file)
        self.n_labels = n_labels
        self.img_size = img_size
        if transform is None:
            transform = ToTensor()
        self.transform = torchvision.transforms.Compose([
            transform, Resize(self.img_size), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), RandomInvert(p = 0.5)
        ])

        # equalize the dataset by provided attribute
        if equalized_by != "none":
            if equalized_by not in ["Race", "Gender", "Age", "Emotion", "Combo"]:
                raise NotImplementedError()

            equalized_by_list = ["Gender", "Race", "Emotion"] if equalized_by == "Combo" else [equalized_by]

            self.dataframe['equalized'] = self.dataframe[equalized_by_list].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
            
            # get counts for the distribution of the attribute
            unique, counts = np.unique(self.dataframe['equalized'],  return_counts=True)
            # how to equalize the counts
            eq_count = None
            if equalized_how == "up":
                eq_count = np.max(counts)
            elif equalized_how == "down":
                eq_count = np.min(counts)
            elif equalized_how == "mean":
                eq_count = int(np.mean(counts))
            else: raise NotImplementedError()

            print(f"Equalizing by {equalized_by_list} how {equalized_how} with {eq_count} counts with {unique} unique")
            # resample each 'class' of the attribute
            sampled_dataframes = []
            for u in unique:
                subset = self.dataframe[self.dataframe['equalized'] == u]
                sampled_dataframe = subset.sample(n=eq_count, replace=len(subset) < eq_count, random_state=42)
                sampled_dataframes.append(sampled_dataframe)
            # make a new dataframe
            new_dataframe = pd.concat(sampled_dataframes)
            self.dataframe = new_dataframe
            self.dataframe.reset_index(inplace=True)

        # compute label distribution
        labels = np.array(self.dataframe["Emotion"])
        self.label_dist = {i: np.sum(labels==i) for i in range(n_labels)}
        
        # compute attribute distribution
        races = np.array(self.dataframe["Race"])
        genders = np.array(self.dataframe["Gender"])
        ages = np.array(self.dataframe["Age"])
        self.attr_counts = np.zeros((3,3,5))
        for r in range(3):
            for g in range(3):
                for a in range(5):
                    self.attr_counts[r,g,a] = np.sum(((races == r) & (genders == g) & (ages == a)))

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
    
    def attr_distribution(self):
        return self.attr_counts
    
class RAFDataset(Dataset):
    def __init__(self, csv_file, n_labels, img_size=224, transform=None, equalized_by="none", equalized_how="none"):
        self.dataframe = pd.read_csv(csv_file)
        self.n_labels = n_labels
        self.img_size = img_size
        if transform is None:
            transform = ToTensor()
        self.transform = torchvision.transforms.Compose([
            transform, Resize(self.img_size), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # equalize the dataset by provided attribute
        if equalized_by != "none":
            if equalized_by not in ["Race", "Gender", "Age", "Emotion", "Combo"]:
                raise NotImplementedError()

            equalized_by_list = ["Gender", "Race", "Emotion"] if equalized_by == "Combo" else [equalized_by]

            self.dataframe['equalized'] = self.dataframe[equalized_by_list].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
            
            # get counts for the distribution of the attribute
            unique, counts = np.unique(self.dataframe['equalized'],  return_counts=True)
            # how to equalize the counts
            eq_count = None
            if equalized_how == "up":
                eq_count = np.max(counts)
            elif equalized_how == "down":
                eq_count = np.min(counts)
            elif equalized_how == "mean":
                eq_count = int(np.mean(counts))
            else: raise NotImplementedError()

            print(f"Equalizing by {equalized_by_list} how {equalized_how} with {eq_count} counts with {unique} unique")
            # resample each 'class' of the attribute
            sampled_dataframes = []
            for u in unique:
                subset = self.dataframe[self.dataframe['equalized'] == u]
                sampled_dataframe = subset.sample(n=eq_count, replace=len(subset) < eq_count, random_state=42)
                sampled_dataframes.append(sampled_dataframe)
            # make a new dataframe
            new_dataframe = pd.concat(sampled_dataframes)
            self.dataframe = new_dataframe
            self.dataframe.reset_index(inplace=True)

        # compute label distribution
        labels = np.array(self.dataframe["Emotion"])
        self.label_dist = {i: np.sum(labels==i) for i in range(n_labels)}
        
        # compute attribute distribution
        races = np.array(self.dataframe["Race"])
        genders = np.array(self.dataframe["Gender"])
        ages = np.array(self.dataframe["Age"])
        self.attr_counts = np.zeros((3,3,5))
        for r in range(3):
            for g in range(3):
                for a in range(5):
                    self.attr_counts[r,g,a] = np.sum(((races == r) & (genders == g) & (ages == a)))

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
    
    def attr_distribution(self):
        return self.attr_counts

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
    
def get_aware_model(model_args):
    """
    Return a model based on model_args.
    Required keys:
        - model_type (must be ResNet18)
        - pretrained
        - n_labels
        - frozen
    """
    if model_args['model_type'] == 'ResNet18':
        return ResNet18Aware(model_args)
    else: raise NotImplementedError()

def load_aware_model(model_path, model_args):
    """
    Load a attribute-aware model from provided path.
    """
    if model_args['model_type'] == 'ResNet18':
        model = ResNet18Aware(model_args)
        model.load_state_dict(T.load(model_path))
        return model
    else: raise NotImplementedError()

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

class ResNet18Aware(T.nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.resnet18 = ResNet18(model_args)
        # remove the head of resnet18
        self.resnet18.fc = T.nn.Sequential()
        # linear layers
        self.linear1 = T.nn.Linear(in_features=3, out_features=512, bias=True)
        self.linear2 = T.nn.Linear(in_features=512, out_features=model_args['n_labels'], bias=True)
    
    def forward(self,x,attr):
        x = self.resnet18(x)
        attr = self.linear1(attr)
        z = x + attr
        out = self.linear2(z)
        return out

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
