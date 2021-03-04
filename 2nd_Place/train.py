import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # uncomment to turn off gpu (see https://stackoverflow.com/a/45773574)
import tensorflow as tf
import time

# if test_mode = True, hold out some training data for validation to test out-of-sample accuracy
# for submission, set test_mode = False
test_mode = False

# load data
solar = pd.read_csv(os.path.join("data", "solar_wind.csv"))
sat_pos = pd.read_csv(os.path.join("data", "satellite_positions.csv"))  # not actually used
dst = pd.read_csv(os.path.join("data", "dst_labels.csv"))
sunspots = pd.read_csv(os.path.join("data", "sunspots.csv"))

# convert timedelta
solar["timedelta"] = pd.to_timedelta(solar["timedelta"])
sat_pos["timedelta"] = pd.to_timedelta(sat_pos["timedelta"])
dst["timedelta"] = pd.to_timedelta(dst["timedelta"])
sunspots["timedelta"] = pd.to_timedelta(sunspots["timedelta"])
sunspots.sort_values(["period", "timedelta"], inplace=True)
sunspots["month"] = list(range(len(sunspots)))
sunspots["month"] = sunspots["month"].astype(int)

# merge sunspots
solar["days"] = solar["timedelta"].dt.days
sunspots["days"] = sunspots["timedelta"].dt.days
solar = pd.merge(
    solar, sunspots[["period", "days", "smoothed_ssn", "month"]], "left", ["period", "days"]
)
solar.drop(columns="days", inplace=True)

# merge labels
solar = pd.merge(solar, dst, "left", ["period", "timedelta"])
solar.sort_values(["period", "timedelta"], inplace=True)
solar.reset_index(inplace=True)

# separate the periods in time for faster calculations
# (This is a remnant of a previous approach, where I was doing a single rolling average calculation
# and wanted to avoid combining different training periods into a single point.
# Later I changed to doing the rolling averages independently for each period, so I think this part is no longer needed.)
solar.loc[solar["period"] != "train_a", "timedelta"] += solar.loc[
    solar["period"] == "train_a", "timedelta"
].max() + pd.to_timedelta("100 days")
solar.loc[solar["period"] == "train_c", "timedelta"] += solar.loc[
    solar["period"] != "train_c", "timedelta"
].max() + pd.to_timedelta("100 days")

# remove weird data (exclude from training and fill for prediction)
solar["bad_data"] = False
solar.loc[solar["temperature"] < 1, "bad_data"] = True
solar.loc[solar["temperature"] < 1, ["temperature", "speed", "density"]] = np.nan
for p in ["train_a", "train_b", "train_c"]:
    curr_period = solar["period"] == p
    solar.loc[curr_period, "train_exclude"] = (
        solar.loc[curr_period, "bad_data"].rolling(60 * 24 * 7, center=False).max()
    )

# fill blanks
solar["month"] = solar["month"].fillna(method="ffill")
train_cols = ["bt", "density", "speed", "bx_gsm", "by_gsm", "bz_gsm", "smoothed_ssn"]
train_short = [c for c in train_cols if c != "smoothed_ssn"]
for p in ["train_a", "train_b", "train_c"]:
    curr_period = solar["period"] == p
    solar.loc[curr_period, "smoothed_ssn"] = (
        solar.loc[curr_period, "smoothed_ssn"]
        .fillna(method="ffill", axis=0)
        .fillna(method="bfill", axis=0)
    )
    roll = (
        solar[train_short].rolling(window=20, min_periods=5).mean().interpolate("linear", axis=0)
    )
    solar.loc[curr_period, train_short] = solar.loc[curr_period, train_short].fillna(roll)
    solar.loc[curr_period, train_short] = (
        solar.loc[curr_period, train_short]
        .fillna(method="ffill", axis=0)
        .fillna(method="bfill", axis=0)
    )

# normalise
norm_df = solar[train_cols].median().to_frame("median")
norm_df["lq"] = solar[train_cols].quantile(0.25)
norm_df["uq"] = solar[train_cols].quantile(0.75)
norm_df["iqr"] = norm_df["uq"] - norm_df["lq"]
norm_df.to_csv("norm_df.csv")
solar[train_cols] = (solar[train_cols] - norm_df["median"]) / norm_df["iqr"]

# interpolate target
solar["dst_interp"] = solar["dst"].interpolate(method="linear")
solar["dst_shift"] = solar["dst_interp"].shift(-60)
solar["dst_shift"] = solar["dst_shift"].fillna(method="ffill")

# aggregate features in 10-minute increments
new_cols = [c + suffix for suffix in ["_mean", "_std"] for c in train_short]
train_cols = new_cols + ["smoothed_ssn"]
mean_cols = [i for i, c in enumerate(train_cols) if "_mean" in c]
new_df = pd.DataFrame(index=solar.index, columns=new_cols)
for p in ["train_a", "train_b", "train_c"]:
    curr_period = solar["period"] == p
    new_df.loc[curr_period] = (
        solar.loc[curr_period, train_short]
        .rolling(window=10, min_periods=1, center=False)
        .agg(["mean", "std"])
        .values
    )
    new_df.loc[curr_period] = new_df.loc[curr_period].fillna(method="ffill").fillna(method="bfill")
solar = pd.concat([solar, new_df], axis=1)
solar[train_cols] = solar[train_cols].astype(float)

assert solar[train_cols + ["dst_interp", "dst_shift"]].isnull().sum().sum() == 0

# sample at 10-minute frequency
solar = solar.iloc[10::10].reset_index()

# data generator
class DataGen(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size, valid_inds, length, shuffle=True):
        self.x = x
        self.y = y
        self.length = length
        self.batch_size = batch_size
        self.valid_inds = np.copy(valid_inds)
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.valid_inds)

    def __get_y__(self):
        return self.y[self.valid_inds]

    def __len__(self):
        return int(np.ceil(len(self.valid_inds) / self.batch_size))

    def __getitem__(self, idx):
        if (idx < self.__len__() - 1) or (len(self.valid_inds) % self.batch_size == 0):
            num_samples = self.batch_size
        else:
            num_samples = len(self.valid_inds) % self.batch_size
        x = np.zeros((num_samples, self.length, self.x.shape[1]))
        y = np.zeros((num_samples,))
        end_indexes = self.valid_inds[idx * self.batch_size : (idx + 1) * self.batch_size]
        for n, i in enumerate(end_indexes):
            x[n] = self.x[i - 24 * 6 * 7 + 1 : i + 1, :]
            y[n] = self.y[i]
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.valid_inds)


# model
inputs = tf.keras.layers.Input((6 * 24 * 7, 13))
conv1 = tf.keras.layers.Conv1D(50, kernel_size=6, strides=3, activation="relu")(inputs)
trim1 = tf.keras.layers.Cropping1D((5, 0))(
    conv1
)  # crop from left so resulting shape is divisible by 6
conv2 = tf.keras.layers.Conv1D(50, kernel_size=6, strides=3, activation="relu")(trim1)
trim2 = tf.keras.layers.Cropping1D((1, 0))(conv2)
conv3 = tf.keras.layers.Conv1D(30, kernel_size=6, strides=3, activation="relu")(trim2)
trim3 = tf.keras.layers.Cropping1D((5, 0))(conv3)
conv4 = tf.keras.layers.Conv1D(30, kernel_size=6, strides=3, activation="relu")(trim3)
conv5 = tf.keras.layers.Conv1D(30, kernel_size=9, strides=9, activation="relu")(conv4)
# extract last data point of previous convolutional layers (crop all but one from left)
comb1 = tf.keras.layers.Concatenate(axis=2)(
    [
        conv5,
        tf.keras.layers.Cropping1D((334, 0))(conv1),
        tf.keras.layers.Cropping1D((108, 0))(conv2),
        tf.keras.layers.Cropping1D((34, 0))(conv3),
        tf.keras.layers.Cropping1D((8, 0))(conv4),
    ]
)
dense = tf.keras.layers.Dense(50, activation="relu")(comb1)
output = tf.keras.layers.Flatten()(tf.keras.layers.Dense(1)(dense))
model = tf.keras.Model(inputs, output)
initial_weights = model.get_weights()
epochs = 2
lr = 0.00025
bs = 32

# cross validation
oos_accuracy = []
valid_bool = solar.index % 6 == 0
np.random.seed(0)
solar["month"] = solar["month"].astype(int)
months = np.sort(solar["month"].unique())
if test_mode:
    holdout_months = np.random.choice(months, len(months) // 5, replace=False)
else:
    holdout_months = []
holdout_bool = solar["month"].isin(holdout_months)
# remove the first week from each period, because not enough data for prediction
holdout_ind_arr = []
valid_ind_arr = []
for p in ["train_a", "train_b", "train_c"]:
    all_p = solar.loc[(solar["period"] == p) & valid_bool].index.values[24 * 7 :]
    all_p_holdout = solar.loc[(solar["period"] == p) & valid_bool & holdout_bool].index.values[
        24 * 7 :
    ]
    holdout_ind_arr.append(all_p_holdout)
    valid_ind_arr.append(all_p)
valid_ind = np.concatenate(valid_ind_arr)
non_exclude_ind = solar.loc[~solar["train_exclude"].astype(bool)].index.values
holdout_ind = np.concatenate(holdout_ind_arr)
holdout_gen = DataGen(
    solar[train_cols].values,
    solar["dst_interp"].fillna(method="ffill").values.flatten(),
    bs,
    holdout_ind,
    6 * 24 * 7,
    False,
)
holdout_preds = np.zeros(len(holdout_ind))
holdout_actuals = holdout_gen.__get_y__()
holdout_accuracy_by_model = []
holdout_accuracy_arr = []
np.random.seed(4)
if test_mode:
    train_months = np.setdiff1d(months, holdout_months)
else:
    train_months = months
np.random.shuffle(months)
num_slices = 5
num_iters = 1
for n in range(num_iters):
    for cv_slice in range(num_slices):
        if (n > 0) and (cv_slice == 0):
            np.random.shuffle(train_months)
        leave_out_months = train_months[
            cv_slice
            * (len(train_months) // num_slices) : (cv_slice + 1)
            * (len(train_months) // num_slices)
        ]
        leave_out_months_ind = solar.loc[
            valid_bool & solar["month"].isin(leave_out_months) & (~holdout_bool)
        ].index.values
        curr_months_ind = solar.loc[
            valid_bool & (~solar["month"].isin(leave_out_months)) & (~holdout_bool)
        ].index.values
        train_ind = np.intersect1d(np.intersect1d(valid_ind, curr_months_ind), non_exclude_ind)
        test_ind = np.intersect1d(valid_ind, leave_out_months_ind)
        train_gen = DataGen(
            solar[train_cols].values,
            solar["dst_interp"].fillna(method="ffill").values.flatten(),
            bs,
            train_ind,
            6 * 24 * 7,
        )
        test_gen = DataGen(
            solar[train_cols].values,
            solar["dst_interp"].fillna(method="ffill").values.flatten(),
            bs,
            test_ind,
            6 * 24 * 7,
        )
        tf.keras.backend.clear_session()
        model = tf.keras.Model(inputs, output)
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
        model.set_weights(initial_weights)
        model.fit(train_gen, validation_data=test_gen, epochs=epochs, verbose=2)
        model.save_weights("model_t_{}.h5".format(n * num_slices + cv_slice))
        oos_accuracy.append(model.evaluate(test_gen, verbose=2)[1])
        print("Out of sample accuracy: ", oos_accuracy)
        print("Out of sample accuracy mean: {}".format(np.mean(oos_accuracy)))
        if test_mode:
            preds = model.predict(holdout_gen)
            holdout_accuracy_by_model.append(np.sqrt(mean_squared_error(holdout_actuals, preds)))
            holdout_preds += preds.flatten()
            print("Holdout accuracy: ", holdout_accuracy_by_model)
            acc = np.sqrt(
                mean_squared_error(
                    holdout_actuals, holdout_preds / (num_slices * n + (cv_slice + 1))
                )
            )
            holdout_accuracy_arr.append(acc)
            print("Holdout accuracy for combined model: {}".format(acc))
        else:
            data_gen = DataGen(
                solar[train_cols].values,
                solar["dst_shift"].fillna(method="ffill").values.flatten(),
                bs,
                train_ind,
                6 * 24 * 7,
            )
            tf.keras.backend.clear_session()
            model = tf.keras.Model(inputs, output)
            model.compile(
                loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                metrics=[tf.keras.metrics.RootMeanSquaredError()],
            )
            model.set_weights(initial_weights)
            model.fit(data_gen, epochs=epochs, verbose=2)
            model.save_weights("model_t_plus_one_{}.h5".format(n * num_slices + cv_slice))
        if test_mode:
            print("Holdout accuracy progression: ", holdout_accuracy_arr)
