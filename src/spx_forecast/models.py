"""Forecasters. Each one owns its scaling so the backtester can never leak."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from . import config
from .features import make_sequences


class Persistence:
    """Tomorrow's return is zero: price stays where it is. The bar to beat."""

    name = "persistence"
    is_sequence = False
    lookback = 1

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class Drift:
    """Constant forecast: the average return of the training window."""

    name = "drift"
    is_sequence = False
    lookback = 1

    def __init__(self):
        self.mu_ = 0.0

    def fit(self, X, y):
        self.mu_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mu_)


class RidgeForecaster:
    """Linear model on the engineered features — a cheap nonlinearity-free control."""

    name = "ridge"
    is_sequence = False
    lookback = 1

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha
        self.scaler_ = StandardScaler()
        self.model_ = Ridge(alpha=alpha)

    def fit(self, X, y):
        self.scaler_.fit(X)
        self.model_.fit(self.scaler_.transform(X), y)
        return self

    def predict(self, X):
        return self.model_.predict(self.scaler_.transform(X))


class LSTMForecaster:
    """Single-layer LSTM over a lookback window of features.

    The network predicts the next-day log return. Training uses a chronological
    validation tail (never a random split) for early stopping.
    """

    name = "lstm"
    is_sequence = True

    def __init__(self, lookback: int = config.LOOKBACK,
                 hidden: int = config.HIDDEN_UNITS, dropout: float = config.DROPOUT,
                 epochs: int = config.MAX_EPOCHS, patience: int = config.PATIENCE,
                 batch_size: int = config.BATCH_SIZE, seed: int = config.SEED,
                 verbose: int = 0):
        self.lookback = lookback
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose
        self.scaler_ = StandardScaler()
        self.model_ = None
        self.y_mu_ = 0.0
        self.y_sigma_ = 1.0

    def fit(self, X, y):
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        tf.keras.utils.set_random_seed(self.seed)
        Xs = self.scaler_.fit_transform(X)
        # Standardise the target too: raw daily returns are ~1e-3, which makes
        # the loss surface badly conditioned and lets a few folds diverge.
        self.y_mu_ = float(np.mean(y))
        self.y_sigma_ = float(np.std(y)) or 1.0
        ys = (y - self.y_mu_) / self.y_sigma_
        seqs, targets = make_sequences(Xs, ys, self.lookback)

        # Chronological 85/15 split for early stopping — the tail is later in time.
        n_val = max(int(0.15 * len(seqs)), 64)
        fit_part, val_part = slice(0, -n_val), slice(-n_val,)

        model = keras.Sequential([
            layers.Input(shape=(self.lookback, Xs.shape[1])),
            layers.LSTM(self.hidden),
            layers.Dropout(self.dropout),
            layers.Dense(1),
        ])
        # Huber keeps one 2020-style crash day from dominating the gradient;
        # clipnorm stops the rare exploding update.
        model.compile(optimizer=keras.optimizers.Adam(3e-4, clipnorm=1.0), loss="huber")
        model.fit(
            seqs[fit_part], targets[fit_part],
            validation_data=(seqs[val_part], targets[val_part]),
            epochs=self.epochs, batch_size=self.batch_size,
            callbacks=[keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.patience,
                restore_best_weights=True)],
            shuffle=False, verbose=self.verbose,
        )
        self.model_ = model
        return self

    def predict(self, X):
        Xs = self.scaler_.transform(X)
        seqs, _ = make_sequences(Xs, np.zeros(len(Xs)), self.lookback)
        raw = self.model_.predict(seqs, verbose=0).ravel()
        pred = raw * self.y_sigma_ + self.y_mu_
        # Safety rail: a forecast beyond 4 training sigmas is an artefact, not a
        # signal. Documented in the README rather than hidden.
        return np.clip(pred, -4 * self.y_sigma_, 4 * self.y_sigma_)


def default_factories() -> dict:
    return {
        "persistence": Persistence,
        "drift": Drift,
        "ridge": RidgeForecaster,
        "lstm": LSTMForecaster,
    }
