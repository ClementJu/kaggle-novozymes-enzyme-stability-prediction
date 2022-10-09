import pandas as pd
from pathlib import Path
from configs import Config, configs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from typing import Union, List
import pickle
import logging


class Dataset:
    def __init__(self, path_to_dataset: Union[Path, str],
                 output_dir: Union[Path, str],
                 val_size: float = 0.2,
                 config_name: str = 'default',
                 path_to_label_encoder: Union[Path, str] = None):
        self.path_to_dataset = Path(path_to_dataset)
        self.output_dir = Path(output_dir)
        self.val_size = val_size

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        self.dataframe = pd.read_csv(self.path_to_dataset)
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X = None
        self.X_seq_id = None
        self.y = None
        self.config = Config(**configs[config_name])

        self.label_encoder = (self._load_label_encoder(path_to_label_encoder=path_to_label_encoder)
                              if path_to_label_encoder is not None
                              else None)

    def apply_training_pipeline(self):
        self.clean()
        self.process()
        self.train_val_split(val_size=self.val_size)
        self.export_training()

    def apply_test_pipeline(self):
        self.process()
        self.export_test()

    def clean(self) -> pd.DataFrame():
        df_clean = self.dataframe.copy()
        cols_to_keep = ['protein_sequence', 'tm', 'seq_id']
        df_clean = df_clean[cols_to_keep]

        nb_elements_before = len(df_clean)
        logging.info(f'> Raw dataset length: {nb_elements_before}')
        df_clean.dropna(inplace=True)
        nb_elements_after = len(df_clean)
        logging.info(f'> {nb_elements_before - nb_elements_after} rows dropped due to null values')

        logging.info(f'> Upper bound for sequence length: {self.config.seq_len_upper_bound_drop}')
        nb_elements_before = len(df_clean)

        if self.config.seq_len_upper_bound_drop is not None:
            df_clean.loc[:, 'sequence_length'] = pd.Series([len(seq) for seq in df_clean.protein_sequence])
            df_clean.drop(df_clean[df_clean.sequence_length < self.config.seq_len_upper_bound_drop].index, inplace=True)
            df_clean.drop(columns='sequence_length', inplace=True)
            nb_elements_after = len(df_clean)
            logging.info(f'> {nb_elements_before - nb_elements_after} rows dropped due to seq_len_upper_bound')

        self.dataframe = df_clean
        logging.info('> Dataset cleaned')

    def process(self) -> None:
        df = self.dataframe.copy()
        max_seq_length_in_df = df.protein_sequence.str.len().max()

        if 'tm' in df.columns:
            self.y = df.tm.to_numpy()

        if self.label_encoder is None:
            # Label encode
            unique_amino_acids = []
            df.protein_sequence.apply(lambda sequence: Dataset.compute_unique_amino_acids(sequence, unique_amino_acids))

            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(unique_amino_acids)

        encoded_sequences = df.protein_sequence.transform(lambda sequence: self.label_encode_sequences(sequence))

        # Padding
        sequence_length = self.config.max_seq_len if self.config.max_seq_len is not None else max_seq_length_in_df

        encoded_sequences = pad_sequences(encoded_sequences, maxlen=sequence_length,
                                          padding='post', truncating='post', value=-1)

        # One-hot encoding
        self.X = to_categorical(encoded_sequences)
        self.X_seq_id = df.seq_id.to_numpy()

        logging.info('> Dataset preprocessed')

    @staticmethod
    def compute_unique_amino_acids(seq: str, accumulator: List[str]) -> None:
        accumulator += [amino_acid for amino_acid in seq if amino_acid not in accumulator]

    def label_encode_sequences(self, seq: str) -> List[int]:
        return self.label_encoder.transform([amino_acid for amino_acid in seq])

    def train_val_split(self, val_size: float, random_state=42):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X,
            self.y,
            test_size=val_size,
            random_state=random_state
        )

    def export_training(self):
        with open(self.output_dir / 'label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)

        with open(self.output_dir / 'X_train.pkl', 'wb') as f:
            pickle.dump(self.X_train, f)

        with open(self.output_dir / 'y_train.pkl', 'wb') as f:
            pickle.dump(self.y_train, f)

        with open(self.output_dir / 'X_val.pkl', 'wb') as f:
            pickle.dump(self.X_val, f)

        with open(self.output_dir / 'y_val.pkl', 'wb') as f:
            pickle.dump(self.y_val, f)

        logging.info(f'> Dataset exported to {self.output_dir}')

    def export_test(self):
        with open(self.output_dir / 'label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)

        with open(self.output_dir / 'X_test.pkl', 'wb') as f:
            pickle.dump(self.X, f)

        with open(self.output_dir / 'X_seq_id.pkl', 'wb') as f:
            pickle.dump(self.X_seq_id, f)

        logging.info(f'> Dataset exported to {self.output_dir}')

    def _load_label_encoder(self, path_to_label_encoder: Union[Path, str]):
        with open(path_to_label_encoder, 'rb') as f:
            return pickle.load(f)
