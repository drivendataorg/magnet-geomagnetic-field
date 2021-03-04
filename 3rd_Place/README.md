# NOAA Challenge 3rd Place Solution

Welcome to the 3rd place solution for the [MagNet: Model the Geomagnetic Field](https://www.drivendata.org/competitions/73/noaa-magnetic-forecasting/) competition.
This repository contains everything needed to replicate our solution. Team LosExtraterrestres 👽👽👽👽

## (0) Getting started

### Prerequisites

You can install all neccesary dependecies on your own machines with conda running the commands below.

```bash
conda env create -f environment.yml
conda activate magnet
```

### Download the Data

First, **download the data** from the competition [download page](https://www.drivendata.org/competitions/73/noaa-magnetic-forecasting/data/)
and put each file in the `data/raw/` folder. After you get the data, you should have
these files:

```text
data/raw/
├── dst_labels.csv
├── satellite_positions.csv
├── solar_wind.csv
└── sunspots.csv
```

## (1) Compute Features

To compute the training dataset, run the following command

```bash
bash commands/compute_dataset.sh --n_jobs {n_jobs}
```

where n_jobs is the number of jobs to run in parallel. the default value is 1

this command will:

- Save the solar wind CSV file as a Feather file.
- Apply the feature engineering pipeline to the time series

after running the commands, you should have these files:

```text
data/interim/
├──  solar_wind.feather
data/processed/
├──  fe.feather
```

this step may take sometime because the solar wind data is very large and we tried to simulate the evaluation process where we predict each timedelta at a time. It is important to use multiple cores to speed things up.

If you want to run the preprocessing on a sample dataset, run the commands below.

```bash
python src/to_feather.py
python src/make_sample.py --frac {frac}
python src/run_fe.py --n_jobs {n_jobs} --use_sample True
```

Where the frac parameter represents the proportion of the dataset to include on the sample dataset.

## (2) Train and Validate Experiments

### Solution Overview

Our solution is an ensemble of 3 models, 1 LGBM and 2 feed-forward neuronets with dropout and batch normalization, you can find the specific parameters of such models in the models_config/ folder. In the case of the LGBM we train 2 models, one for each horizon (t and t + 1 hour) but for the feed-forward neuronet we only train one model.

We compute a lot of features and most of them are redundant, that's why for each model we:

- Train the model with all features
- Calculate the feature importance
- Train the model again but this time only with important features

This approach helps us to reduce overfit, improve our validation score and reduce the complexity of our models.

### Running Experiments

to run all the steps above, run the following command:

```bash
bash commands/train_and_validate.sh
```

to run it on the sample dataset:

```bash
bash commands/train_and_validate.sh --use_sample True
```

for each of the model, this command will:

- Validate the Model using 20% of the data
- Train the final model using all data available

After completing this step, you are ready to do inference!

### Metrics And Artifacts

The scores and parameters of each models are registered using Mlflow, if you want to look at the results open the mlflow web UI with the command below and then open your browser at `http://localhost:<mlflow_port>`

```bash
mlflow ui
```

We save more information about the training and scores, this additional information is saved in the experiment's folder.

### Ensemble

Once you have trained all experiments, the next step is to look how much the score improve by ensembling, We do this by executing the ensemble.py script:

```bash
python src/ensemble.py
```

The script will create a CSV file as below:
|   h0_rmse |   h1_rmse |    rmse | experiment                                |   n_model |
 |----------:|----------:|--------:|:------------------------------------------|----------:|
 |   9.54384 |   9.57683 | 9.56035 | lgbm_1200__neuronet_100__sigmoid_neuronet |         3 |

this CSV file will be save in the following path:

```text
experiments/
├──  ensemble_summary.csv
```
