from tensorflow.keras import losses, metrics
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import LeakyReLU


def create_model_binary(x_size: int) -> Sequential:
    """
    I am creating a neural network with three layers. The first layer has an output
    size of 200, arbitrarily. It has an input dimension equal to x_size, or the number of features
    in x. It has one hidden layer, of size 150 arbitrarily, both of which use relu because
    it is a reasonable thing to use. Finally, we have an output of y_size, which gets a softmax,
    and is compiled with categorical cross-entropy.

    :param x_size: The number of features in x.
    :return: A new model for performing predictions.
    """
    model = Sequential([
        Dense(200, input_dim=x_size),
        LeakyReLU(alpha=0.2),
        Dense(100),
        LeakyReLU(alpha=0.2),
        Dense(50),
        LeakyReLU(alpha=0.2),
        Dense(1, activation='sigmoid')
    ])

    optimizer = Adam(learning_rate=1e-3, name="Adam")
    loss = losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics.BinaryAccuracy()])

    return model


def create_model_multiclass(x_size: int, y_size: int) -> Sequential:
    """
    I am creating a neural network with three layers. The first layer has an output
    size of 200, arbitrarily. It has an input dimension equal to x_size, or the number of features
    in x. It has one hidden layer, of size 150 arbitrarily, both of which use relu because
    it is a reasonable thing to use. Finally, we have an output of y_size, which gets a softmax,
    and is compiled with categorical cross-entropy.

    :param x_size: The number of features in x.
    :param y_size: The number of classes that there could be.
    :return: A new model for performing predictions.
    """
    # We <3 DaTa SciEnce!!
    model = Sequential([
        Dense(200, input_dim=x_size),
        LeakyReLU(alpha=0.2),
        Dense(100),
        LeakyReLU(alpha=0.2),
        Dense(50),
        LeakyReLU(alpha=0.2),
        Dense(y_size, activation='softmax')
    ])

    optimizer = Adam(learning_rate=1e-3, name="Adam")
    loss = losses.CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics.CategoricalAccuracy()])
    return model
