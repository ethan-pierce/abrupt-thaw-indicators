"""Build and train a simple neural network using Keras."""

import tensorflow as tf
from tensorflow import keras

import os
from settings import DATA, MODELS, OUTPUT

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'figure.figsize': (12, 10)})
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

class AbruptThawPredictor:
    """Builds a model of abrupt thaw probability."""

    def __init__(
        self, 
        path_to_data: str,
        test_size: float = 0.2, 
        val_size: float = 0.2,
        initial_bias: float = None
    ):
        """Initialize the model with a path to data and all hyperparameters."""
        self.df = pd.read_csv(path_to_data, index_col = 0)
        self.categorical = [col for col in self.df.columns if self.df[col].isin([0, 1]).all()]
        self.continuous = [col for col in self.df.columns if col not in self.categorical]
        self.categorical.remove('Class')

        self.transformer = ColumnTransformer(
            [
                ('standardize', StandardScaler(), self.continuous),
                ('one-hot', 'passthrough', self.categorical)
            ]
        )

        self.split = self.split_data(test_size, val_size)
        self.training = self.split['training']
        self.validation = self.split['validation']
        self.testing = self.split['testing']

        self.metrics = [
            keras.metrics.BinaryCrossentropy(name = 'cross entropy'),
            keras.metrics.MeanSquaredError(name = 'Brier score'),
            keras.metrics.TruePositives(name = 'tp'),
            keras.metrics.FalsePositives(name = 'fp'),
            keras.metrics.TrueNegatives(name = 'tn'),
            keras.metrics.FalseNegatives(name = 'fn'), 
            keras.metrics.BinaryAccuracy(name = 'accuracy'),
            keras.metrics.Precision(name = 'precision'),
            keras.metrics.Recall(name = 'recall'),
            keras.metrics.AUC(name = 'auc'),
            keras.metrics.AUC(name = 'prc', curve = 'PR'),
        ]

        self.initial_bias = np.log([self.count_classes()[0] / self.count_classes()[1]]) if initial_bias is None else initial_bias
        self.callbacks = self.create_callbacks()

        self.model = self.build_model()

    def count_classes(self) -> tuple:
        """Return a count of the number of each class in the dataset."""
        zeros, ones = np.bincount(self.df["Class"])
        return zeros, ones, zeros + ones

    def split_data(self, test_size: float, val_size: float) -> dict:
        """Split the data into training, validation, and test sets."""
        train_df, test_df = train_test_split(self.df, test_size = test_size)
        train_df, val_df = train_test_split(train_df, test_size = val_size)

        train_labels = train_df.pop('Class')
        test_labels = test_df.pop('Class')
        val_labels = val_df.pop('Class')

        train_std = self.transformer.fit_transform(train_df)
        val_std = self.transformer.transform(val_df)
        test_std = self.transformer.transform(test_df)

        train_array = np.array(train_std)
        val_array = np.array(val_std)
        test_array = np.array(test_std)

        return {
            'training': {'labels': train_labels, 'array': train_array, 'df': train_df
            },
            'validation': {'labels': val_labels, 'array': val_array, 'df': val_df
            },
            'testing': {'labels': test_labels, 'array': test_array, 'df': test_df
            }
        }

    def build_model(self):
        """Build a feedforward neural network in Keras."""
        if self.initial_bias is not None:
            initial_bias = tf.keras.initializers.Constant(self.initial_bias)
        else:
            initial_bias = None

        model = keras.Sequential(
            [   
                keras.layers.Input(shape = (self.training['array'].shape[1],)),
                keras.layers.Dense(16, activation = 'relu'),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(1, activation = 'sigmoid', bias_initializer = initial_bias)
            ]
        )

        model.compile(
            optimizer = keras.optimizers.Adam(learning_rate = 1e-4),
            loss = keras.losses.BinaryCrossentropy(),
            metrics = self.metrics
        )

        return model

    def create_callbacks(self):
        """Create a list of callbacks for the model."""
        return keras.callbacks.EarlyStopping(
            monitor = 'val_prc',
            verbose = 1,
            patience = 10,
            mode = 'max',
            restore_best_weights = True
        )

    def train_model(self, epochs: int, batch_size: int, weights: dict = None):
        """Train the model and return a history of model metrics."""
        history = self.model.fit(
            self.training['array'],
            self.training['labels'],
            batch_size = batch_size,
            epochs = epochs,
            validation_data = (self.validation['array'], self.validation['labels']),
            callbacks = [self.callbacks],
            class_weight = weights
        )

        return history

def plot_metrics(history):
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])

        plt.legend()

def plot_cm(labels, predictions, threshold=0.5):
    cm = sklearn.metrics.confusion_matrix(labels, predictions > threshold)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(threshold))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


if __name__ == '__main__':
    thaw = AbruptThawPredictor(os.path.join(DATA, "clean-feature-table.csv"))

    BATCH_SIZE = 256
    baseline_history = thaw.train_model(epochs = 100, batch_size = BATCH_SIZE)
    baseline_results = thaw.model.evaluate(
        thaw.testing['array'], 
        thaw.testing['labels'], 
        batch_size = BATCH_SIZE, 
        verbose = 0,
        return_dict = True
    )

    for key, val in baseline_results.items():
        print(key, val)

    train_predictions_baseline = thaw.model.predict(thaw.training['array'], batch_size=BATCH_SIZE)
    test_predictions_baseline = thaw.model.predict(thaw.testing['array'], batch_size=BATCH_SIZE)
    
    plot_cm(thaw.testing['labels'], test_predictions_baseline, threshold = 0.5)
    plt.show()
    plot_cm(thaw.testing['labels'], test_predictions_baseline, threshold = 0.1)
    plt.show()
    plot_cm(thaw.testing['labels'], test_predictions_baseline, threshold = 0.9)
    plt.show()

    plot_roc("Train Baseline", thaw.training['labels'], train_predictions_baseline, color=colors[0])
    plot_roc("Test Baseline", thaw.testing['labels'], test_predictions_baseline, color=colors[0], linestyle='--')
    plt.legend(loc='lower right')
    plt.show()

