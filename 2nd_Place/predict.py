import numpy as np
import pandas as pd
import tensorflow as tf

# disable retracing, from: https://stackoverflow.com/a/59943217
tf.compat.v1.disable_eager_execution()

# model
inputs = tf.keras.layers.Input((6*24*7, 13))
conv1 = tf.keras.layers.Conv1D(50, kernel_size=6, strides=3, activation='relu')(inputs)
trim1 = tf.keras.layers.Cropping1D((5, 0))(conv1)
conv2 = tf.keras.layers.Conv1D(50, kernel_size=6, strides=3, activation='relu')(trim1)
trim2 = tf.keras.layers.Cropping1D((1, 0))(conv2)
conv3 = tf.keras.layers.Conv1D(30, kernel_size=6, strides=3, activation='relu')(trim2)
trim3 = tf.keras.layers.Cropping1D((5, 0))(conv3)
conv4 = tf.keras.layers.Conv1D(30, kernel_size=6, strides=3, activation='relu')(trim3)
conv5 = tf.keras.layers.Conv1D(30, kernel_size=9, strides=9, activation='relu')(conv4)
comb1 = tf.keras.layers.Concatenate(axis=2)([conv5,
                                             tf.keras.layers.Cropping1D((334, 0))(conv1),
                                             tf.keras.layers.Cropping1D((108, 0))(conv2),
                                             tf.keras.layers.Cropping1D((34, 0))(conv3),
                                             tf.keras.layers.Cropping1D((8, 0))(conv4)])
dense = tf.keras.layers.Dense(50, activation='relu')(comb1)
output = tf.keras.layers.Flatten()(tf.keras.layers.Dense(1)(dense))
model = tf.keras.Model(inputs, output)
model_t_arr = []
model_t_plus_one_arr = []
num_models = 5
for i in range(num_models):
    new_model = tf.keras.models.clone_model(model)
    new_model.load_weights('model_t_{}.h5'.format(i))
    model_t_arr.append(new_model)
    new_model = tf.keras.models.clone_model(model)
    new_model.load_weights('model_t_plus_one_{}.h5'.format(i))
    model_t_plus_one_arr.append(new_model)

norm_df = pd.read_csv('norm_df.csv', index_col=0)


def predict_dst(
    solar_wind_7d: pd.DataFrame,
    satellite_positions_7d: pd.DataFrame,
    latest_sunspot_number: float,
):
    """
    Take all of the data up until time t-1, and then make predictions for
    times t and t+1.

    Parameters
    ----------
    solar_wind_7d: pd.DataFrame
        The last 7 days of satellite data up until (t - 1) minutes [exclusive of t]
    satellite_positions_7d: pd.DataFrame
        The last 7 days of satellite position data up until the present time [inclusive of t]
    latest_sunspot_number: float
        The latest monthly sunspot number (SSN) to be available

    Returns
    -------
    predictions : Tuple[float, float]
        A tuple of two predictions, for (t and t + 1 hour) respectively; these should
        be between -2,000 and 500.
    """

    solar = solar_wind_7d.copy()
    solar.sort_values('timedelta', inplace=True)
    solar = solar.reset_index()
    solar['smoothed_ssn'] = latest_sunspot_number
    train_cols = ['bt', 'density', 'speed', 'bx_gsm', 'by_gsm', 'bz_gsm', 'smoothed_ssn']

    # remove weird data (will be filled)
    solar.loc[solar['temperature'] < 1, ['temperature', 'speed', 'density']] = np.nan

    # fill blanks
    roll = solar[train_cols].rolling(window=20, min_periods=5).mean().interpolate('linear', axis=0)
    solar[train_cols] = solar[train_cols].fillna(roll)
    solar[train_cols] = solar[train_cols].fillna(method='ffill', axis=0).fillna(method='bfill', axis=0)

    # fill remaining blanks with mean values
    for c in train_cols:
        solar[c] = solar[c].fillna(norm_df.loc[c, 'median'])

    # normalise
    solar[train_cols] = (solar[train_cols] - norm_df['median']) / norm_df['iqr']

    # aggregate features
    train_short = [c for c in train_cols if c != 'smoothed_ssn']
    new_cols = [c + suffix for suffix in ['_mean', '_std'] for c in train_short]
    solar[new_cols] = solar[train_short].rolling(window=10, min_periods=1, center=False)\
        .agg(['mean', 'std']).fillna(method='ffill').fillna(method='bfill').values
    train_cols = new_cols + ['smoothed_ssn']
    solar[train_cols] = solar[train_cols].astype(float)

    assert solar[train_cols].isnull().sum().sum() == 0

    # load model and predict
    pred_data = solar[train_cols].values[-24*60*7 + 9::10, :][np.newaxis, :, :]
    prediction_at_t0 = np.mean([np.array(m.predict(pred_data)).flatten()[0] for m in model_t_arr])
    prediction_at_t1 = np.mean([np.array(m.predict(pred_data)).flatten()[0] for m in model_t_plus_one_arr])

    # restrict to allowed range
    prediction_at_t0 = max(-2000, min(500, prediction_at_t0))
    prediction_at_t1 = max(-2000, min(500, prediction_at_t1))

    return prediction_at_t0, prediction_at_t1
