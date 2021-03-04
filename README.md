[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

![Banner Image](https://drivendata-public-assets.s3.amazonaws.com/noaa-cover-img.png)


# MagNet: Model the Geomagnetic Field

## Goal of the Competition

The efficient transfer of energy from solar wind to Earth’s magnetic field can cause massive geomagnetic storms. The resulting disturbances in the geomagnetic field can wreak havoc on GPS systems, satellite communication, electric power transmission, and more. The Disturbance Storm-Time Index, or Dst, is a measure of the severity of geomagnetic storms. Magnetic surveyors, government agencies, academic institutions, satellite operators, and power grid operators use the Dst index to analyze the strength and duration of geomagnetic storms.

Empirical models have been proposed as early as in 1975 to forecast Dst solely from solar-wind observations at the Lagrangian (L1) position by satellites such as NOAA’s Deep Space Climate Observatory (DSCOVR) or NASA's Advanced Composition Explorer (ACE). Over the past three decades, several models were proposed for solar wind forecasting of Dst, including empirical, physics-based, and machine learning approaches. While the ML models generally perform better than models based on the other approaches, there is still room to improve, especially when predicting extreme events. More importantly, we seek solutions that work on the raw, real-time data streams and are agnostic to sensor malfunctions and noise.

The goal of this challenge is to develop models for forecasting Dst that 1) push the boundary of predictive performance 2) under operationally viable constraints 3) using specified real-time solar-wind data feeds. Competitors were tasked with forecasting both the current Dst value (`t0`), and Dst one hour in the future (`t1`).

## What's in this Repository

This repository contains code from winning competitors in the [MagNet: Model the Geomagnetic Field](https://www.drivendata.org/competitions/73/noaa-magnetic-forecasting/) DrivenData challenge.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

Place | Team or User | Private Score (RMSE) | Public Score (RMSE) | Summary of Model
---   | ---          | ---          | ---           | ---
1     | [Ammarali32](https://www.drivendata.org/users/Ammarali32/) | 11.131 | 11.436 | The proposed model structure consists of a Bidirectional LSTM connected to a Bidirectional GRU. BI-LSTM-GRU is a well known combination for time-series and text recognition problems. The LSTM layers are followed by 3 dense layers connected to the GRU through a flattened layer. I used all features except satellite position data, and left the decision of what features to use and what to ignore to the Neural Network. I aggregated the features in one hour increments. The imputation method that minimized my loss was a simple imputer with most frequent strategy.
2     | [belinda_trotta](https://www.drivendata.org/users/belinda_trotta/)    | 11.253 | 12.013 | I trained a separate ensemble of 5 models for time t and t + 1, so there are 10 models in total. The model is a convolutional neural network with rectified linear activations. The model consists of a set of layers which apply convolutions to detect patterns at progressively longer time spans. Thus the earlier layers recognize lower-level features on short time-spans, and these are aggregated into higher-level patterns spanning longer time ranges in the later layers. I used only the solar wind and sunspots data; I found that the satellite data didn’t help my model. I aggregated the training data in 10-minute increments, taking the mean and standard deviation of each feature in the increment.
3     | LosExtraterrestres: [camaron_ai](https://www.drivendata.org/users/camaron_ai/), [NataliaLaDelMar](https://www.drivendata.org/users/NataliaLaDelMar/)  | 11.294 | 11.537 | Before applying any feature engineering preprocessing, we studied the distribution of the temperature, speed and smoothed_ssn time series. We noticed some degree of skewness on them. In order to reduce it, we applied Box-Cox transformations. We then computed a series of rolling statistical measures over windows of different lengths. For each time series, we computed several different features that are detailed in our report. Finally, we were left with a total of 186 features that we later reduce. Our final solution is an ensemble of 3 models: 1 Gradient Boosting Machine (using the LGBM implementation) and 2 Feed-Forward Neural Nets (NN) with dropout and batch normalization. In the case of the LGBM we train 2 models, one for each horizon (t and t + 1 hour). For the feed-forward NNs, we train only one model. 
4     | k_squared: [KarimAmer](https://www.drivendata.org/users/KarimAmer/), [kareem](https://www.drivendata.org/users/kareem/)  | 11.529 | 12.619 | We preprocessed the “solar_wind” data by aggregating hourly features: mean, std, min, max, and median in addition to the first and last-minute features and the difference between them (gradient). We also added the daily ACE satellite positions to our list of features. We utilize the latest 96 hours (4 days) in our model to predict the following 2 hours. Our model is a 4-block deep convolutional neural network. Each block has two consecutive convolution layers residually connected. The output of the convolutions is passed through Leaky ReLU non-linear activation function then maxpooling to reduce the sequence length by a factor of 2. Finally, a fully-connected layer projects the convolutional features to 2 outputs. We utilize a custom loss function which allows us to control over and under shooting of our loss function. We built an ensemble of models using different power parameters P and different seeds and averaged their predictions.

Additional solution details can be found in the `reports` folder inside the directory for each submission.
