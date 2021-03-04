init_features = ['bx_gsm', 'by_gsm', 'bz_gsm', 'theta_gsm', 'bt',
                 'density', 'speed', 'temperature']
ignore_features = ['timedelta', 'period', 't0', 't1']

default_sample_frac = 0.4
target = ['t0', 't1']
yhat = ['yhat_t0', 'yhat_t1']
keep_columns = ignore_features + yhat
