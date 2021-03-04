import re

_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')


def camel2snake(name):
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()


class CallBack:
    _order = 0
    learn = None
    run_train = True
    run_valid = True

    def __getattr__(self, attr):
        if hasattr(self.learn, attr):
            return getattr(self.learn, attr)
        raise AttributeError(attr)

    def __call__(self, method):
        method_function = getattr(self, method, None)
        _run = ((self.run_train and getattr(self, 'training', True)) or
                (self.run_valid and not getattr(self, 'training', False)))
        return (_run and method_function and method_function())

    @property
    def name(self):
        name = re.sub(r'callback$', '', self.__class__.__name__.lower())
        return camel2snake(name or 'callback')


class TrainEvalCallback(CallBack):
    run_train = True
    _order = -1

    def before_fit(self):
        self.learn.epoch = 0
        self.learn.pct_train = 0.
        self.learn.train_iter = 0
        # self.learn.model.to(device = self.databunch.device)

    def after_batch(self):
        self.learn.pct_train += 1./(self.total_iter*self.n_epoch)
        self.learn.train_iter += 1

    def before_train(self):
        self.learn.pct_train = self.epoch / self.n_epoch
        self.model.train()
        self.learn.training = True

    def before_valid(self):
        self.model.eval()
        self.learn.training = False


class CancelFitException(Exception):
    pass


class CancelTrainException(Exception):
    pass


class CancelValidException(Exception):
    pass


class CancelEpochException(Exception):
    pass

