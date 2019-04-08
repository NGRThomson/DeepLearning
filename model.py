from keras.models import Sequential, Model
from keras.layers import *
from keras import regularizers, initializers
import numpy as np

from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import choice, uniform


def create_model(x_train, y_train, x_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    l2_lambda = 0.0001  # use 0.0001 as a L2-regularisation factor

    model = Sequential()
    model.add(Dense(512, input_shape=(784,), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(l2_lambda)))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([256, 512, 1024])}}, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(l2_lambda)))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0, 1)}}))

    # If we choose 'four', add an additional fourth layer
    if {{choice(['three', 'four'])}} == 'four':
        model.add(Dense(100))

        # We can also choose between complete sets of layers

        model.add({{choice([Dropout(0.5), Activation('linear')])}})
        model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    result = model.fit(x_train, y_train,
              batch_size={{choice([64, 128])}},
              epochs=2,
              verbose=1,
              validation_split=0.1)
    #get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history['val_acc'])
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

# def get_model():
#     kernel_size = 3  # we will use 3x3 kernels throughout
#     pool_size = 2  # we will use 2x2 pooling throughout
#     conv_depth = 32  # use 32 kernels in both convolutional layers
#     drop_prob_1 = 0.25  # dropout after pooling with probability 0.25
#     drop_prob_2 = 0.5  # dropout in the FC layer with probability 0.5
#     hidden_size = 128  # there will be 128 neurons in both hidden layers
#     l2_lambda = 0.0001  # use 0.0001 as a L2-regularisation factor
#     ens_models = 3  # we will train three separate models on the data
#     height, width, depth = 28, 28, 1
#     inp = Input(shape=(height, width, depth))  # N.B. TensorFlow back-end expects channel dimension last
#     inp_norm = BatchNormalization()(inp)  # Apply BN to the input (N.B. need to rename here)
#
#     outs = []  # the list of ensemble outputs
#     for i in range(ens_models):
#         # Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer), applying BN in between
#         conv_1 = Convolution2D(conv_depth, (kernel_size, kernel_size),
#                                padding='same',
#                                kernel_initializer='he_uniform',
#                                kernel_regularizer=l2(l2_lambda),
#                                activation='relu')(inp_norm)
#         conv_1 = BatchNormalization()(conv_1)
#         conv_2 = Convolution2D(conv_depth, (kernel_size, kernel_size),
#                                padding='same',
#                                kernel_initializer='he_uniform',
#                                kernel_regularizer=l2(l2_lambda),
#                                activation='relu')(conv_1)
#         conv_2 = BatchNormalization()(conv_2)
#         pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
#         drop_1 = Dropout(drop_prob_1)(pool_1)
#         flat = Flatten()(drop_1)
#         hidden = Dense(hidden_size,
#                        kernel_initializer='he_uniform',
#                        kernel_regularizer=l2(l2_lambda),
#                        activation='relu')(flat)  # Hidden ReLU layer
#         hidden = BatchNormalization()(hidden)
#         drop = Dropout(drop_prob_2)(hidden)
#         outs.append(Dense(10,
#                           kernel_initializer='glorot_uniform',
#                           kernel_regularizer=l2(l2_lambda),
#                           activation='softmax')(drop))  # Output softmax layer
#
#     out = average(outs)  # average the predictions to obtain the final output
#
#     model = Model(inputs=inp, outputs=out)  #
#     return model


if __name__ == '__main__':

    model = get_model()
