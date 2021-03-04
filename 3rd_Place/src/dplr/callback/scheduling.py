import numpy as np
import math
from typing import Callable, Union, List
from matplotlib import pyplot as plt
from dplr.callback import CallBack, CancelFitException


def display_sched(schedule):
    plt.plot([schedule(_x) for _x in np.linspace(0, 1, 100)])
    plt.show()


class Anneler:
    def __init__(self, function: Callable,
                 start: Union[float, int], end: Union[float, int]):
        self.end = end
        self.start = start
        self.function = function

    def __call__(self, position):
        return self.function(position=position, start=self.start, end=self.end)

    def __repr__(self):
        return f'{self.function.__name__}(start={self.start},end={self.end})'

    def display(self):
        return display_sched(self)


def linear_schedule(start, end, position):
    return start + position*(end - start)


def exponential_schedule(start, end, position):
    return start * (end/start) ** position


def cosine_schedule(start, end, position):
    amplitude = (end - start)/2
    return start + (1 + math.cos(math.pi*(1-position))) * amplitude


def no_schedule(position, start, end):
    return start


def NoSchedule(lr):
    return Anneler(no_schedule, lr, lr)


def CosineSchedule(start, end):
    return Anneler(cosine_schedule, start, end)


def LinearSchedule(start, end):
    return Anneler(linear_schedule, start, end)


def ExpSchedule(start, end):
    return Anneler(exponential_schedule, start, end)


def CombineScheduler(pcts: List[float], schedulers: List[Anneler]):
    assert len(pcts) == len(schedulers), \
           'the lenght of the pcts and schedulers must match!'
    assert sum(pcts) == 1, 'the pcts must sum to 1'
    pcts = [0.] + pcts
    pcts = np.cumsum(pcts)

    def _scheduler(position):
        for boundary in range(len(pcts) - 1):
            lower_limit = pcts[boundary]
            upper_limit = pcts[boundary+1]

            if lower_limit <= position <= upper_limit:
                scaled_position = (position-lower_limit) / (upper_limit-lower_limit)
                return schedulers[boundary](scaled_position)
    return _scheduler


class ParamScheduler(CallBack):
    run_valid = False

    def __init__(self, function: Callable, schedule: Callable):
        self.function = function
        self.schedule = schedule

    def _update_value(self, position):
        self.function(self.learn, self.schedule(position))

    def before_batch(self):
        self._update_value(self.pct_train)

    @property
    def name(self):
        return self.function.__name__


def lr_update(learner, value):
    for param in learner.optimizer.param_groups:
        param['lr'] = value


class LRFinder(ParamScheduler):
    _order = 100

    def __init__(self, start: float = 1e-5, end: float = 10.,
                 n_iter: int = 100):
        schedule = ExpSchedule(start, end)
        super().__init__(lr_update, schedule)
        self.n_iter = n_iter

    def before_batch(self):
        super()._update_value(self.learn.n_iter / self.n_iter)

    def after_batch(self):
        if self.learn.n_iter > self.n_iter:
            raise CancelFitException


# def find_lr(learner,  start = 1e-5, end = 1, n_iter = 100, scheduler = ExpSchedule):
#     schedule = LRFinder(start, end, n_iter)
#     callbacks = [schedule, Recoder(), ProgressBarCallBack()]
#     learner.fit(1, cbs= callbacks)