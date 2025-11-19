import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset
import  PIL.Image as Image
import torchvision

def plot_classes(set:Subset, class_names:list[str], set_name:str)->None:

    img_per_class = dict(zip(class_names, np.zeros(len(class_names))))
    for i,class_num in set:
         img_per_class[class_names[class_num]] += 1

    plt.figure(figsize=(20, 8))
    plt.bar(list(class_names), img_per_class.values(), color='g')
    plt.title(f"Images per class for {set_name} set")
    plt.xticks(rotation=90)
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.show()



def plot_random_six_images(set, class_names:list[str])->None:
    idx = np.random.choice(np.arange(start=0, stop=len(set)), size=6, replace=False)
    # idx = [1000, 10000, 15000, 20000, 200, 300]
    fig, axes = plt.subplots(2,3,figsize=(20, 8))
    for i, k in enumerate(idx):
        img, class_num = set[k]

        img_show = Image.fromarray(np.array(img))
        img_show = img_show.convert('RGB')
        ax = axes[i//3][np.mod(i,3)]
        ax.imshow(img_show)
        ax.set_xlabel(class_names[class_num])

    return idx

def plot_idx_labeled(idx, preds, dataset):
    fig, axes = plt.subplots(2,3,figsize=(20, 8))
    fig.subplots_adjust(top=1)

    for i, k in enumerate(idx):
        img, class_num = dataset[k]

        img_show = Image.fromarray(np.array(img))
        img_show = img_show.convert('RGB')
        ax = axes[i//3][np.mod(i,3)]
        ax.imshow(img_show)
        ax.set_xlabel(f"Predicted_label: {preds[i]}\n"
                      f"True_label: {dataset.classes[class_num]}")

    return idx


def plot_loss(train_loss:list[float], val_loss: list[float]) ->None:
    plt.plot(train_loss, label="Train Loss", color='blue')
    plt.plot(val_loss, label="Validation Loss", color='red')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_time(epoch_time:list[float])->None:
    plt.plot(epoch_time, label="Epoch Time")
    plt.title('Epoch Time')
    plt.xlabel('Epoch')
    plt.ylabel('Time')
    plt.show()

def plot_with_function(idx, preds, dataset, func, transform):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fig, axes = plt.subplots(2,3,figsize=(20, 8))
    fig.subplots_adjust(top=1)

    for i, k in enumerate(idx):
        img, class_num = dataset[k]
        img_tensor = (transform(img).unsqueeze(0)).cpu()
        img_tensor.requires_grad_(True)

        # Calculating Grad Cam
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            attribution = func(img_tensor, target=int(preds[i][0]))

        attribution_np = attribution.detach().cpu().numpy()
        attribution_np = np.max(np.abs(attribution_np), axis=0)


        min_val = np.min(attribution_np)
        max_val = np.max(attribution_np)

        if max_val > min_val:
            attribution_np = (attribution_np - min_val) / (max_val - min_val)

        attribution_show = attribution_np[0]

        # Getting and resizng orginal image
        original_img, _ = dataset[k]
        img_show = Image.fromarray(np.array(original_img))
        img_show = img_show.convert('RGB')
        img_show = img_show.resize(size=(224,224))


        ax = axes[i//3][np.mod(i,3)]
        ax.imshow(img_show)
        at = ax.imshow(attribution_show, cmap='plasma',alpha=0.7, interpolation='none', vmin=0, vmax=attribution_show.max())
        ax.set_xlabel(f"Predicted_label: {dataset.classes[preds[i][0]]}\n"
                      f"True_label: {dataset.classes[class_num]}")