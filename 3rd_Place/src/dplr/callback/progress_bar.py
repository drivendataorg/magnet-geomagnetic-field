from dplr.callback import CallBack
from tqdm.auto import tqdm


class ProgressBarCallBack(CallBack):
    def before_fit(self):
        self.master_bar = tqdm(range(self.n_epoch), desc='epochs')
        train_lenght = len(self.databunch.train_dl)
        self.iter_bar = tqdm(range(train_lenght), desc='iterations')
        # self.valid_step = round(len(self.databunch.train_dl)
        # / len(self.databunch.valid_dl))
        # print(self.valid_step)

    def after_epoch(self):
        self.master_bar.update()

    def before_train(self):
        self.iter_bar.reset()

    def after_fit(self):
        self.master_bar.close()
        self.iter_bar.close()

    def after_batch(self):
        if self.training:
            self.iter_bar.update()
            self.iter_bar.set_postfix({'loss': self.batch['loss'].item()})

    def after_train(self):
        self.iter_bar.refresh()
