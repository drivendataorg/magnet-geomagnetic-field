from dplr.callback import CallBack, CancelFitException, CancelTrainException
from dplr.callback import CancelValidException, CancelEpochException
from dplr.callback import TrainEvalCallback
from torch import nn, optim
from typing import List
import torch
from dplr.data import ConcatDict


def seed_everything(seed: int):
    torch.manual_seed(seed)
    # torch.set_deterministic(True)


class Learner:
    default_callbacks = [TrainEvalCallback]
    training = True

    def __init__(self, model: nn.Module,
                 optimizer: optim.Optimizer,
                 databunch, callbacks: List[CallBack] = []):
        self.model = model
        self.optimizer = optimizer
        self.databunch = databunch
        self.callbacks = []
        self.add_callbacks([(cb() if isinstance(cb, type) else cb)
                            for cb in callbacks + self.default_callbacks])
        self('after_init')

    def add_callbacks(self, callbacks):
        for callback in callbacks:
            self._add_callback(callback)

    def remove_callbacks(self, callbacks):
        for callback in callbacks:
            self._remove_callback(callback)

    def _add_callback(self, callback):
        # check if exists
        old_callback = getattr(self, callback.name, None)
        assert old_callback is None, 'callback is already registered'
        callback.learn = self
        self.callbacks.append(callback)
        setattr(self, callback.name, callback)

    def _remove_callback(self, callback):
        # unlink the learner from the callback
        callback.learn = None
        if callback in self.callbacks:
            # remove it from the callbacks
            self.callbacks.remove(callback)
        # remove the learn attr
        if hasattr(self, callback.name):
            delattr(self, callback.name)

    def _with_events(self, function, event_name, execption):
        try:
            self(f'before_{event_name}')
            function()
        except execption:
            self(f'after_cancel_{event_name}')
        finally:
            self(f'after_{event_name}')

    def _do_one_batch(self) -> None:
        output = self.model(**self.batch)
        self.batch.update(output)
        self('after_pred')
        if not self.training:
            return None
        self('before_backward')
        self.batch['loss'].backward()
        self('after_backward')
        self.optimizer.step()
        self('after_step')
        self.optimizer.zero_grad()

    def do_one_batch(self, n_iter, batch):
        self.batch = batch
        self.n_iter = n_iter
        self._with_events(self._do_one_batch, 'batch', CancelFitException)

    def all_batches(self):
        self.total_iter = len(self.dl)
        for n_iter, batch in enumerate(self.dl):
            self.do_one_batch(n_iter, batch)

    def _do_train_epoch(self):
        self.dl = self.databunch.train_dl
        self.model.train()
        self._with_events(self.all_batches, 'train', CancelTrainException)

    def _do_valid_epoch(self):
        self.dl = (self.databunch.valid_dl
                   if self.databunch.valid_dl is not None else
                   self.databunch.train_dl)
        self.model.eval()
        self._with_events(self.all_batches, 'valid', CancelValidException)

    def do_one_epoch(self):
        self._do_train_epoch()
        self._do_valid_epoch()

    def _do_fit(self):
        for epoch in range(self.n_epoch):
            self.epoch = epoch
            self._with_events(self.do_one_epoch, 'epoch', CancelEpochException)

    def fit(self, epochs: int = 1, cbs: List[CallBack] = [],
            seed: int = None):
        self.n_epoch = epochs
        self.add_callbacks(cbs)
        if seed is not None:
            seed_everything(seed)
        self._with_events(self._do_fit, 'fit', CancelFitException)
        self.reset_attr()

    def reset_attr(self):
        self.batch, self.epoch, self.dl, self.n_iter = None, None, None, None

    def __call__(self, event_name):
        callbacks = [callback for callback in
                     sorted(self.callbacks, key=lambda cb: cb._order)
                     if hasattr(callback, event_name)]
        for callback in callbacks:
            callback(event_name)


def predict_dl(model, dl):
    model.eval()
    with torch.no_grad():
        return ConcatDict([model(**batch) for batch in dl])
