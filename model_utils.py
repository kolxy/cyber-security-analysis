from tensorflow.keras import losses, metrics
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam


def create_model(x_size: int, y_size: int) -> Sequential:
    """
    I am creating a neural network with three layers. The first layer has an output
    size of 200, arbitrarily. It has an input dimension equal to x_size, or the number of features
    in x. It has one hidden layer, of size 150 arbitrarily, both of which use relu because
    it is a reasonable thing to use. Finally, we have an output of y_size, which gets a softmax,
    and is compiled with binary cross-entropy.

    :param x_size: The number of features in x.
    :param y_size: The number of classes that there could be.
    :return: A new model for performing predictions.
    """

    model = Sequential([
        Dense(50, input_dim=x_size, activation='relu', kernel_initializer='uniform'),
        Dense(100, activation='relu', kernel_initializer='uniform'),
        Dense(y_size, activation='softmax', kernel_initializer='uniform')
    ])

    optimizer = Adam(learning_rate=1e-3)
    loss = losses.CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics.CategoricalAccuracy(),
                                                           metrics.Precision(),
                                                           metrics.Recall()])
    return model