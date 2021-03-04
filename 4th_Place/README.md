# Predicting Disturbance-Storm-Time Index (DST) using Ensemble of 1D-CNNs with Residual Connections

Fourth place solution for MagNet: Model the Geomagnetic Field.

## Summary of Approach

We preprocessed the “solar_wind” data by aggregating hourly features: mean, std, min, max, and median in addition to the first and last-minute features and the difference between them (gradient). We also added the daily ACE satellite positions. We utilize the latest 96 hours (4 days) in our model to predict the following 2 hours.

Our model is a 4-block deep convolutional neural network. Each block has two consecutive convolution layers residually connected. The output of the convolutions is passed through Leaky ReLU non-linear activation function then maxpooling to reduce the sequence length by a factor of 2. Finally, a fully-connected layer projects the convolutional features to 2 outputs.

We utilize a custom loss function: Loss = log((y-p)^2)^p+|y-p|
where the logarithm is raised to a power P. This allows us to control over and under shooting of our loss function. We built an ensemble of models using different power parameters P and different seeds and averaged their predictions.

## Prerequisites

Firstly, you need to have 

* Ubuntu 18.04 
* Python 3.6.9
* 32G RAM
* 11 GB GPU RAM

Secondly, you need to install the dependencies by running:

```
pip3 install -r requirements.txt
```

## Project files

* train.py: trains an ensemble of 21 models and saves their weights.
* validate.py: validates an ensemble of 21 models using two-stage time-split k-fold cross validation
* predict.py: has predict function for DrivenData testing environment.
* model.py: has the CNN model class.
* dataset.py: has the dataset class for testing environment.
* conv*: model weights used in our best submission.
* scaler.pck: object with saved normalization parameters. 
* config.json: model configuration paramters.

## Running

### Training

```
python3 train.py --data path (fill it with the data path on your machine)
```

### Validation

```
python3 validate.py --data path (fill it with the data path on your machine)
```
