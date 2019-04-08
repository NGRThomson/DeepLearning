from data import load_data
from model import create_model
from hyperas import optim
from keras.layers import *
from keras.models import Sequential, Model
from keras import regularizers, initializers
import numpy as np

from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import choice, uniform

def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    (x_train, y_train, x_test, y_test) = load_data()

    return x_train, y_train, x_test, y_test



if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=1,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    best_model.summary()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
