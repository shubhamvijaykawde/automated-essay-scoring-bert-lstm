import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop


def build_lstm_model(input_dim: int) -> Sequential:
    """
    Builds and compiles the LSTM model.
    """
    model = Sequential([
        LSTM(
            300,
            dropout=0.4,
            recurrent_dropout=0.4,
            return_sequences=True,
            input_shape=(1, input_dim)
        ),
        LSTM(64, recurrent_dropout=0.4),
        Dropout(0.5),
        Dense(1, activation="relu")
    ])

    model.compile(
        loss="mean_squared_error",
        optimizer=RMSprop(),
        metrics=["mae"]
    )

    return model


def reshape_for_lstm(X: np.ndarray) -> np.ndarray:
    """
    Reshapes 2D vectors to LSTM-compatible 3D tensors.
    """
    return X.reshape(X.shape[0], 1, X.shape[1])
