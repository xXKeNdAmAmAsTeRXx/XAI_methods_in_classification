import os
import copy
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd


def test_model(model_name: str, model:torch.nn, loader: DataLoader, class_name:list[str]) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name())

    model.to(device)
    model.eval()
    preds = []
    trues = []

    for img, label in loader:
        img = img.to(device)
        output = model(img)

        pred = output.detach().cpu().numpy().argmax(axis=1)
        true = label.cpu().numpy()

        preds.append(pred)
        trues.append(true)

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    print(f'accuracy: {accuracy_score(trues, preds)}')

    cf_mtx = confusion_matrix(trues, preds)

    fig, ax = plt.subplots(figsize=(16, 16))
    im = ax.imshow(cf_mtx, interpolation='nearest', cmap='viridis')
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.5)

    ax.set(
        title=f"Confusion Matrix for {model_name}",
        ylabel="True label",
        xlabel="Predicted label"
    )
    plt.show()

    # cr = classification_report(trues, preds, target_names=class_name, labels=np.arange(len(class_name)))
    # print(cr)

    cr_dict = classification_report(
        trues,
        preds,
        target_names=class_name,
        labels=np.arange(len(class_name)),
        output_dict=True
    )

    df = pd.DataFrame(cr_dict).T

    return df



def test_on_given_idx(dataset, net_model, transform, idx):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_model.to(device)
    net_model.eval()

    dataset_2 = copy.deepcopy(dataset)
    dataset_idx = Subset(dataset_2, idx)
    dataset_idx.dataset.transform = transform
    loader = DataLoader(dataset_idx, batch_size=1, shuffle=False, num_workers=0)

    preds = []

    Data = {
        'Idx': [],
        'True_Label': [],
        'Predicted_Label': []
    }
    for i, (img, label) in enumerate(loader):
        Data['Idx'].append(idx[i])

        img_tensor = img.to(device)
        output = net_model(img_tensor)
        pred = output.detach().cpu().numpy().argmax(axis=1)
        pred_name = dataset.classes[pred[0]]

        Data['True_Label'].append(dataset.classes[label.numpy()[0]])
        Data['Predicted_Label'].append(pred_name)
        preds.append(pred_name)

    return preds, pd.DataFrame(Data)