# Real_or_Fake

## Introduction
Given an image, classify if the image is real or product of tampering. Worked with 32000 high resolution images. On literature survey, found out a technique called [Error Level Analysis](https://resources.infosecinstitute.com/error-level-analysis-detect-image-manipulation/#gref) which is being used by forensics to determine tampering with photo evidences. ELA is based on uniformity of lossy compression techniques. 
In our approach, we applied ELA transformation to images. We implemented a multi modal in-model concatenation pipeline for classification task. The intuition of this napproach is that the model will be able to compare the features extracted from NNet trained on ELA transformed images with that of original images and learn the specific patterns representative of tampering.
Below is a summary
![summary](/image/summary.png)

## Requirements
1. fastai
2. joblib
3. pandas
4. numpy
5. pytorch
6. torch

## Architecture explained 
The multi modal in-model concatenation architechture has two parallel running pipeline as shown in the image below:
![cnn-based-architecture](/image/cnn-architecture.png)
There are two blocks in the architecture
#### Block 1: Multi Modal
- part 1: Fine tuning pretrained ResNet101 i.e. transfer learning
- part 2: Training the entire network
- part 3: Improve models by increasing the resolution of input images
- part 4: Extracting values of Fully Connected layer and concatenating to form one array
Above parts are completed for both ELA transformed images and Original images
#### Block 2: In-model concatenation
- part 1: Using a two layerd perceptron based NNet with two output units. The FC NNet takes concatenated output from FC layers of trained ResNets. Features from a fully connected layer is extracted from both models (ELA transformed ResNet101 and Original Image ResNet101)
- part 2: Concatenated feature array is used as input to predict probability of an image being 'Real' or 'Fake(tampered)'

## On-going work
Literature survey to assess siamese network[https://arxiv.org/pdf/1709.08761.pdf] for the classification task
Below is the architecture:
![Siamese-Network](/image/siamese-network.png)

