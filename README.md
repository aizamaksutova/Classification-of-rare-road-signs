# Classification_of_rare_road_signs
Project on building a classifier for rare traffic signs

## Overview
This repository addresses the classification of rare road signs, a challenge due to the scarcity of real-world examples. The project introduces synthetic sampling to create data for these uncommon signs, enhancing classification methods. Classifying rare road signs is difficult because of limited real-world examples. Traditional classification struggles with new and uncommon sign types.

## Solution. What do we do to tackle this problem? 
Synthetic Sample Generation: Generating synthetic examples of rare signs to augment the dataset.
Specialized Classifier: Implementing a nearest neighbor method classifier optimized for these synthetic samples.

## Goals
Demonstrate synthetic sample generation for rare road sign classification.
Test a tailored classifier and improve key metrics.

# Implementation process
First, to start the process, get all the utils functions you need to run

```
chmod a+x scripts/build.sh
./build
python3 run.py
```

## Synthetic data generationg for upsampling the road-signs dataset

For upsampling we use the dataset in class DatasetRTSD for loading the data and class SignGenerator to generate data when training further models

## Classification models

### Simple classifier based on a pre-trained neural network

Implement a simple classifier based on a pre-trained neural network (e.g., ResNet-50) in the CustomNetwork class. It is necessary to replace the last full-link layer in the neural network by a linear layer with internal_features parameters. Next, added the ReLU activation function and a full-link layer for classification with the required number of classes (205) as a separate parameter. Also implemented the train_simple_classifier() function to train the classifier

### A more advanced classifier 

In this clause, I will build a special classifier that works better for rare signs in this task. It consists of two separate parts:
- A specially trained neural network for classifying traffic signs. It will additionally introduce a separate loss function for the features generated in the penultimate layer of the network.
- Nearest neighbor method for classifying road signs based on the features of the penultimate layer of the neural network.
The general scheme of operation is described as follows. A sign is taken, run through the neural network, and the features from the penultimate layer are extracted. The classification is performed using the nearest neighbors method.

The model uses contrastive loss as main metric, which can be calculated with the following formula
![loss](https://github.com/aizamaksutova/Classification-of-rare-road-signs/blob/main/misc/img_loss.png)

All the implementations can be seen in the rare_traffic_sign_solution.py file
