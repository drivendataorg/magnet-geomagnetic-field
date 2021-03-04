from typing import Dict, Callable, List
from collections import defaultdict
import pandas as pd
from dplr.callback import CallBack
import numpy as np


def record_lr(learner):
    return learner.optimizer.param_groups[-1]['lr']


def record_loss(learner):
    return learner.batch['loss'].item()


def record_wd(learner):
    return learner.optimizer.param_groups[-1]['weight_decay']


class Recoder(CallBack):
    run_valid = False
    defaults = {'lr': record_lr, 'loss': record_loss, 'wd': record_wd}

    def __init__(self, record_function: Dict[str, Callable] = {}):
        self.data = defaultdict(list)
        self.defaults.update(record_function)
        self.record_function = self.defaults
        for item in self.record_function.keys():
            setattr(self, item, self.data[item])

    def after_batch(self):
        for item, function in self.record_function.items():
            self.data[item].append(function(self))

    @property
    def get_data(self):
        return pd.DataFrame(self.data)


class Metric:
    def __init__(self, function: Callable, reduce: str = 'mean'):
        self.function = function
        self.reset()
        self.reduce = reduce

    @property
    def name(self):
        if hasattr(self.function, '__name__'):
            return self.function.__name__
        elif hasattr(self.function, '__class__'):
            return self.function.__class__.__name__
        return 'Metric'

    def reset(self):
        self._accumulated_scores = []
        self.size = 0

    def __call__(self,  yhat, y):
        score = self.function(yhat, y)
        if hasattr(score, 'detach'):
            score = score.detach()
        score = float(score)
        if self.reduce == 'sum':
            score *= len(y)
        self._accumulated_scores.append(score)
        self.size += len(y)
        return score

    @property
    def average(self):
        reduce = getattr(self, 'reduce', 'mean')
        if reduce == 'sum':
            return np.sum(self._accumulated_scores) / self.size
        return np.mean(self._accumulated_scores)


class MetricRecorderCallBack(CallBack):
    _order = 1

    def __init__(self, *metrics: List[Callable], reduce: str = 'mean'):
        self._metrics = [Metric(function, reduce=reduce)
                         for function in metrics]
        self.train_metrics = defaultdict(list)
        self.valid_metrics = defaultdict(list)
        self._metrics_name = ['loss'] + [metric.name
                                         for metric in self._metrics]

    def _reset_metrics(self):
        for metric in self._metrics:
            metric.reset()
        self._loss = []

    def before_train(self):
        self._reset_metrics()

    def before_valid(self):
        self._reset_metrics()

    def _record_metrics(self, recoder):
        recoder['loss'].append(np.mean(self._loss))
        for metric in self._metrics:
            recoder[metric.name].append(metric.average)

    def after_train(self):
        self._record_metrics(self.train_metrics)

    def after_valid(self):
        self._record_metrics(self.valid_metrics)

    def after_pred(self):
        self._loss.append(self.batch['loss'].item())
        for metric in self._metrics:
            metric(self.batch['prediction'], self.batch['target'])

    def _get_epoch_data(self, metrics):
        return {metric_name: scores[self.epoch]
                for metric_name, scores in metrics.items()}

    def after_epoch(self):
        print('train', self._get_epoch_data(self.train_metrics),
              'valid', self._get_epoch_data(self.valid_metrics))

    def after_fit(self):
        self.learn.metrics_table = self.metrics_table

    @property
    def metrics_table(self):
        train_metrics = pd.DataFrame(self.train_metrics)
        train_metrics.columns = [f'train_{name}'
                                 for name in self._metrics_name]
        valid_metrics = pd.DataFrame(self.valid_metrics)
        valid_metrics.columns = [f'valid_{name}'
                                 for name in self._metrics_name]
        return pd.concat([train_metrics, valid_metrics], axis=1)
