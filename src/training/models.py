from keras.callbacks import EarlyStopping, CSVLogger
from keras.layers import Input, Dense, Dropout, Flatten, Activation
from keras.layers import Conv1D, Add, MaxPooling1D, BatchNormalization
from keras.layers import Embedding, Bidirectional, CuDNNLSTM, GlobalMaxPooling1D
from keras.regularizers import l2
from keras.models import Model, Sequential
from keras.metrics import MeanSquaredError
from typing import Tuple, List, Union
from pathlib import Path
import matplotlib.pyplot as plt
from keras.layers import LSTM
import tensorflow as tf
from configs import BiLstmConfig
import logging
import pickle
import pandas as pd


class BaseModel:
    def __init__(self, X_train, y_train, X_val, y_val, model, config):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model = model
        self.training_history = None
        self.metric = MeanSquaredError()
        self.config = config

    def score_self(self):
        self._score_helper(self.X_train, self.y_train, 'Training')
        self._score_helper(self.X_val, self.y_val, 'Validation')

    def score(self, X, y, dataset_name: str):
        self._score_helper(X, y, dataset_name)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def _score_helper(self, X, y, dataset_name=''):
        score_value = self.model.evaluate(X, y, batch_size=self.config.batch_size, verbose=1)
        logging.info(f'> {dataset_name} > Loss: {score_value[0]}, {self.metric}: {score_value[1]}')

    def summary(self):
        self.model.summary()

    def export(self, output_dir: Union[str, Path]):
        output_dir = BaseModel.create_output_dir(output_dir=output_dir)
        self.model.save_weights(output_dir / 'model.h5')

    def apply_prediction_pipeline(self, path_to_model: Union[Path, str],
                                  path_to_test_set: Union[Path, str],
                                  path_to_ids: Union[Path, str],
                                  path_output_dir: Union[Path, str]):
        path_output_dir = BaseModel.create_output_dir(path_output_dir)
        self.load_model(path_to_model=path_to_model)
        self.generate_predictions(path_to_test_set=path_to_test_set, path_to_ids=path_to_ids, path_output_dir=path_output_dir)

    def load_model(self, path_to_model: Union[Path, str]):
        path_to_model = Path(path_to_model)
        self.model.load_weights(path_to_model)

    def generate_predictions(self,
                             path_to_test_set: Union[Path, str],
                             path_to_ids: Union[Path, str],
                             path_output_dir: Union[Path, str]):

        path_to_test_set = Path(path_to_test_set)
        path_output_dir = BaseModel.create_output_dir(output_dir=path_output_dir)

        with open(path_to_test_set, 'rb') as f:
            X_test = pickle.load(f)

        import IPython; IPython.embed(); exit()
        preds = self.predict(X_test)
        preds = preds.reshape(len(preds))   # Reshape to 1D-array

        with open(path_to_ids, 'rb') as f:
            ids = pickle.load(f)

        submission_df = pd.DataFrame.from_dict({"seq_id": ids, "tm": preds})
        submission_df.to_csv(path_output_dir / 'predictions.csv', index=False)

    def plot(self, output_dir: Union[str, Path]):
        plt.figure(figsize=(12, 5))

        metric_names = list(filter(lambda m: not m.startswith('val_'), self.training_history.history))

        for index, metric_name in enumerate(metric_names):
            plt.subplot(1, len(metric_names), index + 1)

            metric_value = self.training_history.history[metric_name]
            metric_value_validation = self.training_history.history[f'val_{metric_name}']

            x_axis = range(0, len(metric_value))

            plt.plot(x_axis, metric_value, label=f'Training {metric_name}')
            plt.plot(x_axis, metric_value_validation, label=f'Validation {metric_name}')
            plt.title(f'Training and validation {metric_name}')
            plt.legend()

        output_dir = BaseModel.create_output_dir(output_dir=output_dir)
        plt.savefig(output_dir / 'graphs.png')

    @staticmethod
    def create_output_dir(output_dir: Union[str, Path]):
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        return output_dir


class BidirectionalLSTM(BaseModel):
    def __init__(self, X_train, y_train, X_val, y_val, config_name: str):
        config = BiLstmConfig.get_config(config_name)
        BaseModel.__init__(self,
                           X_train, y_train, X_val, y_val,
                           self._build_model(config),
                           config)

    def fit(self, output_dir: Union[str, Path]):
        self.summary()
        output_dir = BaseModel.create_output_dir(output_dir=output_dir)
        self.training_history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_val, self.y_val),
            callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=1),
                       CSVLogger(output_dir / 'log.csv', append=True, separator=';')],
            **self.config.as_fit_config_dict()
        )

    def _build_model(self, config):
        units = 20
        model = Sequential()
        model.add(Bidirectional(LSTM(units=units, return_sequences=True), input_shape=(222, 20,)))
        model.add(Bidirectional(LSTM(units=units)))
        model.add(Dense(1, activation='linear'))

        opt = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

        model.compile(loss='mse', optimizer=opt, metrics=[MeanSquaredError()])
        return model

    @classmethod
    def for_prediction(cls):
        return cls(None, None, None, None, 'default')
