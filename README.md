# Intro
This repository contains a gradCAM method in explainable AI.


## Dataset
The Stanford Dogs dataset contains images of 120 breeds of dogs from around the world. This dataset has been built using images and annotation from ImageNet for the task of fine-grained image categorization. It was originally collected for fine-grain image categorization, a challenging problem as certain dog breeds have near identical features or differ in colour and age.

![img.png](assets/dogs.png)

From kaggle: https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset <br/>
Originally from http://vision.stanford.edu/aditya86/ImageNetDogs/


## ResNet
![img_1.png](assets/resnet.png)
ResNet architecture is Convolutional Neural Network (CNN) architecture in which layers learn residual function with reference to the layer inputs.
In this repo I decided to use ResNet18(the smallest version of ResNets) as model performance optimization is not the main purpose. 

More about ResNet in following paper: https://arxiv.org/abs/1512.03385


