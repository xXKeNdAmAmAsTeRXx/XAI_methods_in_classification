import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset
import  PIL.Image as Image

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
    print(idx)
    fig, axes = plt.subplots(2,3,figsize=(20, 8))
    for i, k in enumerate(idx):
        img, class_num = set[k]

        img_show = Image.fromarray(np.array(img))
        img_show = img_show.convert('RGB')
        ax = axes[i//3][np.mod(i,3)]
        ax.imshow(img_show)
        ax.set_xlabel(class_names[class_num])

