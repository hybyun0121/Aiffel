import math
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

class CosineAnnealingScheduler(Callback):
    '''
    Cosine annealing scheduler
    input:
        T_max:
        eta_max: 최대 learning rate
        eta_min: 최소 learning rate
        verbose:
    output:

    hasattr(object, name) : object에 name attribute가 있는지 확인하는 함수

    '''

    def __init__(self, T_max, eta_max, eta_min=0, verbose=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'learning_rate'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.eta_min + (self.eta_max - self.eta_min)*(1+math.cos(math.pi*epoch/self.T_max))/2
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d : CosineAnnealingScheduler setting learning'
            'rate to %s.' %(epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
