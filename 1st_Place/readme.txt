This is the first place submission.

Training:

To regenerate the weights and other files upload this notebook to Google colab. Then make sure that the data is accessible by colab and run the notebook. It will take about 2 hours to finish training. Then you will be able to download the submission files (model, predict.py, scaler, imputer, and config).

Prediction:

Add all files together in a submission archive and use the NOAA repo for inference and testing.

If you are going to use Google Colab then everything will be fine, otherwise make sure that you have all the requirements specified in requirements.txt.



Note from DrivenData Reviewer: In verifying this solution, we found two bugs in the training code. First, the labels used to train are behind by one hour, i.e. the column labeled `t0` is actually `t-1`, and the column labeled `t1` is actually `t0`. The second bug is related to the test/validation split. As currently implemented, the validation set is a subset of the test set, instead of an altogether separate sample. We have left the code as-is so that the first-place solution may be reproduced exactly. However, we have left detailed comments in-line so that the user may update the code as desired.
