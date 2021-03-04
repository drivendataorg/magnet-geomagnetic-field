# Saving solar wind CSV file as a Feather file.
echo "save the solar wind CSV file as a Feather file."
python src/to_feather.py;
# Applying the feature engineering pipeline to the time series
echo "Applying the feature engineering pipeline to the time series"
python src/run_fe.py "$@";