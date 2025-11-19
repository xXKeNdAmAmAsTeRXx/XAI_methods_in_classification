import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from torch.utils.data import DataLoader
import pandas as pd


def test_model(model_name: str, model, loader: DataLoader, class_name):
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
    im = ax.imshow(cf_mtx, interpolation='nearest', cmap='grey')
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.5)

    ax.set(
        title="Confusion Matrix",
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